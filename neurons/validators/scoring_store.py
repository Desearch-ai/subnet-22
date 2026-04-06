import asyncio
import hashlib
import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Mapping, Sequence

import bittensor as bt
import jsonpickle

from neurons.validators.scoring_dataset import SCORING_SEARCH_TYPES, ScoringAssignment
from neurons.validators.storage import (
    ObjectStorage,
    StorageObject,
    build_validator_bucket_name,
)

STORAGE_VERSION = 1
READ_CONCURRENCY = 20


class ScoringStore:
    def __init__(
        self,
        *,
        object_storage: ObjectStorage,
        netuid: int,
        validator_uid: int,
        validator_hotkey: str,
    ):
        self.object_storage = object_storage
        self.netuid = int(netuid)
        self.validator_uid = int(validator_uid)
        self.validator_hotkey = validator_hotkey
        self.bucket_name = build_validator_bucket_name(
            netuid=self.netuid,
            validator_uid=self.validator_uid,
            hotkey=self.validator_hotkey,
        )
        self.bucket_locator = self.object_storage.build_bucket_locator(
            bucket=self.bucket_name
        )
        self.bucket_ref = self.object_storage.resolve_bucket_locator(
            self.bucket_locator
        )

    def _bucket_name_for(self, validator_uid: int, validator_hotkey: str) -> str:
        return build_validator_bucket_name(
            netuid=self.netuid,
            validator_uid=validator_uid,
            hotkey=validator_hotkey,
        )

    def _assignment_payload(self, assignment: ScoringAssignment) -> dict[str, Any]:
        return {
            "time_range_start": assignment.time_range_start.astimezone(
                timezone.utc
            ).isoformat(),
            "uid": int(assignment.uid),
            "search_type": assignment.search_type,
            "validator_uid": int(assignment.validator_uid),
            "validator_hotkey": assignment.validator_hotkey,
            "query": assignment.question.query,
            "params": assignment.question.params,
            "scoring_seed": int(assignment.scoring_seed),
        }

    def _location_for_assignment(self, assignment: ScoringAssignment) -> StorageObject:
        return self._location_for_assignment_with_bucket(
            assignment,
            bucket=self._bucket_name_for(
                assignment.validator_uid,
                assignment.validator_hotkey,
            ),
        )

    def _location_for_assignment_with_bucket(
        self,
        assignment: ScoringAssignment,
        *,
        bucket: str,
    ) -> StorageObject:
        query_hash = hashlib.sha256(
            json.dumps(
                {
                    "query": assignment.question.query,
                    "params": assignment.question.params,
                },
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()[:16]
        window_ts = int(
            assignment.time_range_start.astimezone(timezone.utc).timestamp()
        )
        return StorageObject(
            bucket=bucket,
            key=(
                f"windows/{window_ts}/{assignment.search_type}/"
                f"uid-{assignment.uid}-{assignment.scoring_seed}-{query_hash}.json"
            ),
        )

    async def save_response(
        self,
        assignment: ScoringAssignment,
        response: Any,
    ) -> None:
        if assignment.validator_uid != self.validator_uid:
            raise ValueError(
                "Cannot save a response for another validator bucket: "
                f"expected uid={self.validator_uid}, got uid={assignment.validator_uid}."
            )
        if assignment.validator_hotkey != self.validator_hotkey:
            raise ValueError(
                "Cannot save a response for another validator hotkey: "
                f"expected hotkey={self.validator_hotkey}, "
                f"got hotkey={assignment.validator_hotkey}."
            )

        location = self._location_for_assignment_with_bucket(
            assignment,
            bucket=self.bucket_ref,
        )
        payload = {
            "version": STORAGE_VERSION,
            "assignment": self._assignment_payload(assignment),
            "response": jsonpickle.encode(response),
        }

        await self.object_storage.put_object(
            bucket=location.bucket,
            key=location.key,
            data=json.dumps(payload, sort_keys=True).encode("utf-8"),
            content_type="application/json",
        )

    def _decode_response_payload(
        self,
        assignment: ScoringAssignment,
        raw_payload: bytes | None,
    ) -> dict[str, Any] | None:
        if raw_payload is None:
            return None

        try:
            payload = json.loads(raw_payload.decode("utf-8"))
        except (TypeError, ValueError) as exc:
            bt.logging.warning(
                "[ScoringStore] Failed to decode stored response "
                f"uid={assignment.uid} type={assignment.search_type}: {exc}"
            )
            return None

        if payload.get("version") != STORAGE_VERSION:
            bt.logging.warning(
                "[ScoringStore] Ignoring stored response with unexpected version "
                f"uid={assignment.uid} type={assignment.search_type}"
            )
            return None

        if payload.get("assignment") != self._assignment_payload(assignment):
            bt.logging.warning(
                "[ScoringStore] Ignoring stored response with mismatched assignment "
                f"uid={assignment.uid} type={assignment.search_type}"
            )
            return None

        try:
            response = jsonpickle.decode(payload["response"])
        except Exception as exc:
            bt.logging.warning(
                "[ScoringStore] Failed to deserialize stored response "
                f"uid={assignment.uid} type={assignment.search_type}: {exc}"
            )
            return None

        return {
            "uid": assignment.uid,
            "response": response,
            "scoring_seed": assignment.scoring_seed,
        }

    async def _load_assignment(
        self,
        assignment: ScoringAssignment,
        bucket_locators: Mapping[tuple[int, str], str],
        semaphore: asyncio.Semaphore,
    ) -> tuple[str, dict[str, Any] | None]:
        locator = bucket_locators.get(
            (assignment.validator_uid, assignment.validator_hotkey)
        )
        if not locator:
            bt.logging.warning(
                "[ScoringStore] Missing bucket locator for validator "
                f"uid={assignment.validator_uid} hotkey={assignment.validator_hotkey}"
            )
            return assignment.search_type, None

        try:
            bucket_ref = self.object_storage.resolve_bucket_locator(locator)
        except Exception as exc:
            bt.logging.warning(
                "[ScoringStore] Invalid bucket locator for validator "
                f"uid={assignment.validator_uid} hotkey={assignment.validator_hotkey}: {exc}"
            )
            return assignment.search_type, None

        location = self._location_for_assignment_with_bucket(
            assignment,
            bucket=bucket_ref,
        )
        async with semaphore:
            raw_payload = await self.object_storage.get_object(
                bucket=location.bucket,
                key=location.key,
            )

        if raw_payload is None:
            bt.logging.warning(
                "[ScoringStore] Missing stored response "
                f"uid={assignment.uid} type={assignment.search_type} "
                f"validator={assignment.validator_uid}"
            )

        return assignment.search_type, self._decode_response_payload(
            assignment,
            raw_payload,
        )

    async def get_all_for_assignments(
        self,
        assignments: Sequence[ScoringAssignment],
        *,
        bucket_locators: Mapping[tuple[int, str], str],
    ) -> Dict[str, List[Dict]]:
        if not assignments:
            return {}

        semaphore = asyncio.Semaphore(READ_CONCURRENCY)
        tasks = [
            asyncio.create_task(
                self._load_assignment(assignment, bucket_locators, semaphore)
            )
            for assignment in assignments
        ]

        results = await asyncio.gather(*tasks)

        grouped: Dict[str, List[Dict]] = {key: [] for key in SCORING_SEARCH_TYPES}
        for search_type, item in results:
            if item is not None:
                grouped[search_type].append(item)

        return {
            search_type: sorted(items, key=lambda item: item["uid"])
            for search_type, items in grouped.items()
            if items
        }
