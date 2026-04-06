import json
from datetime import datetime, timezone

import pytest

from neurons.validators.service.scoring_dataset import (
    ScoringAssignment,
    ScoringQuestion,
)
from neurons.validators.service.scoring_store import ScoringStore
from neurons.validators.storage.local import LocalObjectStorage


def build_assignment(epoch_start: datetime) -> ScoringAssignment:
    return ScoringAssignment(
        time_range_start=epoch_start,
        uid=11,
        search_type="web_search",
        validator_uid=5,
        validator_hotkey="validator-a",
        question=ScoringQuestion(
            query="what is bittensor",
            params={"country": "us"},
        ),
        scoring_seed=123,
    )


@pytest.mark.asyncio
async def test_scoring_store_round_trip_local_storage(tmp_path):
    store = ScoringStore(
        object_storage=LocalObjectStorage(tmp_path),
        netuid=22,
        validator_uid=5,
        validator_hotkey="validator-a",
    )
    epoch_start = datetime(2026, 4, 6, 10, 0, tzinfo=timezone.utc)
    assignment = build_assignment(epoch_start)
    response = {"query": "what is bittensor", "results": ["tao"]}

    await store.save_response(assignment, response)

    all_responses = await store.get_all_for_assignments(
        [assignment],
        bucket_locators={(5, "validator-a"): store.bucket_locator},
    )

    assert all_responses == {
        "web_search": [
            {
                "uid": 11,
                "response": response,
                "scoring_seed": 123,
            }
        ]
    }


@pytest.mark.asyncio
async def test_scoring_store_ignores_mismatched_assignment_payload(tmp_path):
    storage = LocalObjectStorage(tmp_path)
    store = ScoringStore(
        object_storage=storage,
        netuid=22,
        validator_uid=5,
        validator_hotkey="validator-a",
    )
    epoch_start = datetime(2026, 4, 6, 10, 0, tzinfo=timezone.utc)
    assignment = build_assignment(epoch_start)
    location = store._location_for_assignment(assignment)
    payload = {
        "version": 1,
        "assignment": {
            **store._assignment_payload(assignment),
            "query": "different question",
        },
        "response": json.dumps({"query": "different question"}),
    }

    await storage.put_object(
        bucket=location.bucket,
        key=location.key,
        data=json.dumps(payload).encode("utf-8"),
    )

    assert (
        await store.get_all_for_assignments(
            [assignment],
            bucket_locators={(5, "validator-a"): store.bucket_locator},
        )
        == {}
    )
