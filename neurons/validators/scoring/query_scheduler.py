import asyncio
import math
import random
import time
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Dict, Optional

import bittensor as bt
import numpy as np

from neurons.validators.scoring import capacity, miner_db
from neurons.validators.scoring.constants import (
    COVERAGE_EXPONENT,
    DEFAULT_PER_UID,
    MIN_VOLUME_RATIO,
    QUALITY_EXPONENT,
    QUALITY_THRESHOLDS,
    SEARCH_TYPE_WEIGHTS,
    VOLUME_EXPONENT,
)
from neurons.validators.scoring.scoring_store import SEARCH_TYPES, ScoringStore
from neurons.validators.scoring.synthetic_query_generator import SyntheticQueryGenerator

ORGANIC_VALUE_MULTIPLIER = 3
ORGANIC_DEEP_CAP_PER_TYPE = 100

BATCH_FRACTION = 0.50
BATCH_INTERVAL_SECONDS = 3
GROUP_SIZE = 2

DEEP_SAMPLE_RATE = 0.20
DEEP_SAMPLE_FLOOR = 1
DEEP_SAMPLE_WEIGHT = 5


def combine_superlinear_scores(
    qualities_per_type: dict[str, dict[int, tuple[float, int]]],
) -> dict[int, float]:
    all_uids: set[int] = set()
    for uid_q in qualities_per_type.values():
        all_uids.update(uid_q.keys())

    combined: dict[int, float] = {}
    for uid in all_uids:
        qv = {
            t: qualities_per_type.get(t, {}).get(uid, (0.0, 0))
            for t in SEARCH_TYPE_WEIGHTS
        }
        max_v = max(v for _, v in qv.values())

        served: dict[str, float] = {}
        for t, (q, v) in qv.items():
            if max_v == 0 or q < QUALITY_THRESHOLDS[t]:
                served[t] = 0.0
            else:
                served[t] = min(1.0, (v / max_v) / MIN_VOLUME_RATIO)

        coverage = sum(SEARCH_TYPE_WEIGHTS[t] * served[t] for t in SEARCH_TYPE_WEIGHTS)

        per_type_sum = 0.0
        for t, (q, v) in qv.items():
            if served[t] == 0.0:
                continue
            per_type_sum += (
                SEARCH_TYPE_WEIGHTS[t]
                * served[t]
                * q**QUALITY_EXPONENT
                * v**VOLUME_EXPONENT
            )

        combined[uid] = coverage**COVERAGE_EXPONENT * per_type_sum

    return combined


class QueryScheduler:
    """
    Background scheduler that drives scoring queries using locally-generated
    synthetic questions.

    Lifecycle per UTC hour:
      1. Batch-generate all queries for every active UID via SyntheticQueryGenerator.
         Epoch-level params (tools, date_filter) are shared; only question text varies.
         Each miner gets N queries per search type where N = verified concurrency.
      2. Dispatch shuffled UID groups in 50% per-UID bursts.
      3. Save the miner's response in ScoringStore.
      4. On hour boundary -> score the previous hour's responses and update capacity.

    Organic responses collected during the epoch are also loaded and a
    capped-random sample is deep-scored. Organic rewards carry
    ``ORGANIC_DEEP_SCORE_WEIGHT`` weight in the per-UID mean.

    Each validator generates its own synthetics independently.
    """

    SPREAD_SECONDS = 55 * 60  # Spread queries over 55 minutes of each hour
    MIN_DISPATCH_WINDOW_SECONDS = 15 * 60  # Below this, skip dispatch + scoring

    def __init__(
        self,
        neuron,
        generator: SyntheticQueryGenerator,
        scoring_store: ScoringStore,
        validators: Dict,  # {"ai_search": ..., "x_search": ..., "web_search": ...}
    ):
        self.neuron = neuron
        self.generator = generator
        self.scoring_store = scoring_store
        self.validators = validators

    def _extract_prompt(self, response) -> str:
        if isinstance(response, dict):
            for key in ("prompt", "query", "content", "id"):
                value = response.get(key)
                if value:
                    return str(value)
            urls = response.get("urls")
            if urls:
                return ", ".join(str(url) for url in urls)
            return ""

        for key in ("prompt", "query", "content", "id"):
            value = getattr(response, key, None)
            if value:
                return str(value)

        urls = getattr(response, "urls", None)
        if urls:
            return ", ".join(str(url) for url in urls)

        return ""

    async def _send_and_save(
        self,
        search_type: str,
        uid: int,
        query: dict,
        time_range_start: datetime,
    ) -> None:
        """Send one scoring query to a specific miner and persist the response."""
        try:
            validator = self.validators[search_type]
            response = await validator.send_scoring_query(query, uid=uid)
            if response is not None:
                await self.scoring_store.save_synthetic(
                    time_range_start, uid, search_type, response
                )
                bt.logging.debug(
                    f"[QueryScheduler] Saved response uid={uid} type={search_type}"
                )
        except Exception as e:
            bt.logging.error(
                f"[QueryScheduler] Scoring query failed uid={uid} type={search_type}: {e}"
            )

    async def _dispatch_epoch(
        self,
        items: list,
        time_range_start: datetime,
    ) -> None:
        """Dispatch each type through shuffled UID groups."""
        for search_type in SEARCH_TYPES:
            if self._current_hour_start() != time_range_start:
                return

            grouped: dict[int, list] = defaultdict(list)
            for item in items:
                if item["search_type"] == search_type:
                    grouped[item["uid"]].append(item)

            shuffled_uids = sorted(grouped)
            random.shuffle(shuffled_uids)
            bt.logging.info(
                f"[QueryScheduler] Phase {search_type}: "
                f"{sum(len(v) for v in grouped.values())} queries "
                f"across {len(shuffled_uids)} shuffled UIDs "
                f"in groups of {GROUP_SIZE}"
            )

            for start in range(0, len(shuffled_uids), GROUP_SIZE):
                if self._current_hour_start() != time_range_start:
                    return
                group = shuffled_uids[start : start + GROUP_SIZE]
                await asyncio.gather(
                    *[
                        self._dispatch_uid(
                            uid, search_type, grouped[uid], time_range_start
                        )
                        for uid in group
                    ],
                    return_exceptions=True,
                )

    @staticmethod
    def _batch_size_for_uid(uid_items: list) -> int:
        return max(1, math.ceil(len(uid_items) * BATCH_FRACTION))

    async def _dispatch_uid(
        self,
        uid: int,
        search_type: str,
        uid_items: list,
        time_range_start: datetime,
    ) -> None:
        batch_size = self._batch_size_for_uid(uid_items)
        batches = [
            uid_items[i : i + batch_size]
            for i in range(0, len(uid_items), batch_size)
        ]

        for batch_idx, batch in enumerate(batches):
            if self._current_hour_start() != time_range_start:
                return

            if batch_idx > 0:
                await asyncio.sleep(BATCH_INTERVAL_SECONDS)

            await asyncio.gather(
                *[
                    self._send_and_save(
                        search_type, uid, item["query"], time_range_start
                    )
                    for item in batch
                ],
                return_exceptions=True,
            )

    def _sample_deep_synth(self, synth_items: list) -> set[int]:
        """Pick DEEP_SAMPLE_RATE of each UID's synth items (floor DEEP_SAMPLE_FLOOR)."""
        by_uid: dict[int, list[int]] = defaultdict(list)
        for idx, item in enumerate(synth_items):
            by_uid[item["uid"]].append(idx)
        sampled: set[int] = set()
        for indices in by_uid.values():
            n = max(DEEP_SAMPLE_FLOOR, round(len(indices) * DEEP_SAMPLE_RATE))
            sampled.update(random.sample(indices, min(n, len(indices))))
        return sampled

    def _sample_organic_deep(self, organic_items: list) -> set[int]:
        """Allocate ORGANIC_DEEP_CAP_PER_TYPE deep slots across UIDs proportional
        to their organic count (largest-remainder), then pick that many at random
        from each UID's organics."""
        if len(organic_items) <= ORGANIC_DEEP_CAP_PER_TYPE:
            return set(range(len(organic_items)))

        by_uid: dict[int, list[int]] = defaultdict(list)
        for idx, item in enumerate(organic_items):
            by_uid[item["uid"]].append(idx)

        total = len(organic_items)
        cap = ORGANIC_DEEP_CAP_PER_TYPE
        quotas_float = {uid: cap * len(idxs) / total for uid, idxs in by_uid.items()}
        quotas = {uid: int(q) for uid, q in quotas_float.items()}
        leftover = cap - sum(quotas.values())
        if leftover > 0:
            ordered = sorted(
                by_uid,
                key=lambda u: quotas_float[u] - quotas[u],
                reverse=True,
            )
            for uid in ordered[:leftover]:
                quotas[uid] += 1

        sampled: set[int] = set()
        for uid, n in quotas.items():
            if n > 0:
                sampled.update(random.sample(by_uid[uid], min(n, len(by_uid[uid]))))
        return sampled

    async def _run_full_scoring(
        self,
        validator,
        items: list,
        time_range_start: datetime,
    ) -> np.ndarray:
        if not items:
            return np.zeros(0, dtype=np.float32)
        responses = [item["response"] for item in items]
        uids = np.array([item["uid"] for item in items], dtype=np.int64)
        prompts = [self._extract_prompt(r) for r in responses]
        result = await validator.compute_rewards_and_penalties(
            event={},
            prompts=prompts,
            responses=responses,
            uids=uids,
            start_time=time.time(),
            scoring_epoch_start=time_range_start,
        )
        if result is None:
            return np.zeros(len(items), dtype=np.float32)
        return np.asarray(result[0], dtype=np.float32)

    async def _score_one_type(
        self,
        search_type: str,
        synthetics: dict,
        organics: dict,
        time_range_start: datetime,
        window_start: str,
        allocations: dict[int, int],
    ) -> dict[int, tuple[float, int]]:
        """Score synth + organic for one type and update capacity per UID.

        Four buckets, each carrying its own weight in the per-UID mean:
            synth-cheap   = 1
            synth-deep    = DEEP_SAMPLE_WEIGHT
            organic-cheap = ORGANIC_VALUE_MULTIPLIER
            organic-deep  = ORGANIC_VALUE_MULTIPLIER * DEEP_SAMPLE_WEIGHT

        Synthetics: 20% per-UID deep sample, cheap on the rest. Organics: code
        checks on all, ORGANIC_DEEP_CAP_PER_TYPE deep slots distributed across
        UIDs proportional to their organic count."""
        validator = self.validators.get(search_type)
        if validator is None:
            return {}

        synth_items = synthetics.get(search_type, [])
        organic_items = organics.get(search_type, [])

        if not synth_items and not organic_items:
            return {}

        deep_synth_idx = self._sample_deep_synth(synth_items)
        deep_synth = [item for i, item in enumerate(synth_items) if i in deep_synth_idx]
        cheap_synth = [
            item for i, item in enumerate(synth_items) if i not in deep_synth_idx
        ]

        deep_organic_idx = self._sample_organic_deep(organic_items)
        deep_organic = [
            item for i, item in enumerate(organic_items) if i in deep_organic_idx
        ]
        cheap_organic = [
            item for i, item in enumerate(organic_items) if i not in deep_organic_idx
        ]

        bt.logging.info(
            f"[QueryScheduler] {search_type}: "
            f"synth={len(synth_items)} (deep={len(deep_synth)}, cheap={len(cheap_synth)}), "
            f"organic={len(organic_items)} (deep={len(deep_organic)}, cheap={len(cheap_organic)})"
        )

        uid_totals: dict[int, float] = defaultdict(float)
        uid_weights: dict[int, float] = defaultdict(float)
        uid_volumes: dict[int, int] = defaultdict(int)

        cheap_items = cheap_synth + cheap_organic
        if cheap_items:
            try:
                cheap_scores = await validator.compute_cheap_scores(
                    [item["response"] for item in cheap_items],
                    np.array([item["uid"] for item in cheap_items], dtype=np.int64),
                )
            except Exception as e:
                bt.logging.error(
                    f"[QueryScheduler] Cheap scoring failed {search_type}: {e}"
                )
                cheap_scores = np.zeros(len(cheap_items), dtype=np.float32)
            scores = cheap_scores.tolist()
            for i, item in enumerate(cheap_synth):
                uid = item["uid"]
                uid_totals[uid] += scores[i]
                uid_weights[uid] += 1
                uid_volumes[uid] += 1
            offset = len(cheap_synth)
            for i, item in enumerate(cheap_organic):
                uid = item["uid"]
                uid_totals[uid] += ORGANIC_VALUE_MULTIPLIER * scores[offset + i]
                uid_weights[uid] += ORGANIC_VALUE_MULTIPLIER
                uid_volumes[uid] += 1

        deep_items = deep_synth + deep_organic
        if deep_items:
            try:
                full_scores = await self._run_full_scoring(
                    validator, deep_items, time_range_start
                )
            except Exception as e:
                bt.logging.error(
                    f"[QueryScheduler] Full scoring failed {search_type}: {e}"
                )
                full_scores = np.zeros(len(deep_items), dtype=np.float32)
            scores = full_scores.tolist()
            for i, item in enumerate(deep_synth):
                uid = item["uid"]
                uid_totals[uid] += DEEP_SAMPLE_WEIGHT * scores[i]
                uid_weights[uid] += DEEP_SAMPLE_WEIGHT
                uid_volumes[uid] += 1
            offset = len(deep_synth)
            organic_deep_weight = ORGANIC_VALUE_MULTIPLIER * DEEP_SAMPLE_WEIGHT
            for i, item in enumerate(deep_organic):
                uid = item["uid"]
                uid_totals[uid] += organic_deep_weight * scores[offset + i]
                uid_weights[uid] += organic_deep_weight
                uid_volumes[uid] += 1

        uid_results = {
            uid: (uid_totals[uid] / uid_weights[uid], uid_volumes[uid])
            for uid in uid_totals
            if uid_weights[uid] > 0
        }

        for uid, (quality, _) in uid_results.items():
            await capacity.record_window_quality(
                uid=uid,
                search_type=search_type,
                quality=quality,
                window_start=window_start,
                allocated=allocations.get(uid, DEFAULT_PER_UID),
            )
        return uid_results

    async def _dispatch_combined_scores(self, combined: dict[int, float]) -> None:
        """Push the combined per-UID scores into the neuron's EMA."""
        if not combined:
            return
        uids_array = np.array(list(combined.keys()), dtype=np.int64)
        rewards_array = np.array(list(combined.values()), dtype=np.float32)
        await self.neuron.update_moving_averaged_scores(uids_array, rewards_array)

    async def score_epoch(
        self,
        time_range_start: datetime,
        allocations_by_type: dict[str, dict[int, int]],
    ) -> None:
        """Load all responses for a completed hour and run reward/penalty
        computation. ``allocations_by_type`` is the per-UID synthetic budget
        that was active during this epoch — captured by the caller before
        the next epoch's ``bulk_update_verified`` overwrites it."""
        try:
            bt.logging.info(
                f"[QueryScheduler] Scoring epoch {time_range_start.isoformat()}"
            )
            synthetics = await self.scoring_store.get_synthetics_for_range(
                time_range_start
            )
            organics = await self.scoring_store.get_organics_for_range(time_range_start)

            if not synthetics and not organics:
                bt.logging.warning(
                    f"[QueryScheduler] No responses for epoch "
                    f"{time_range_start.isoformat()}, skipping scoring."
                )
                return

            window_start = time_range_start.isoformat()
            qualities_per_type: dict[str, dict[int, tuple[float, int]]] = {}

            for search_type in SEARCH_TYPES:
                uid_results = await self._score_one_type(
                    search_type,
                    synthetics,
                    organics,
                    time_range_start,
                    window_start,
                    allocations_by_type.get(search_type, {}),
                )
                if uid_results:
                    qualities_per_type[search_type] = uid_results

            touched_uids = sorted(
                {uid for results in qualities_per_type.values() for uid in results}
            )
            await capacity.ramp_after_epoch(touched_uids)

            combined = combine_superlinear_scores(qualities_per_type)
            await self._dispatch_combined_scores(combined)

        except Exception as e:
            bt.logging.error(f"[QueryScheduler] Error in score_epoch: {e}")

    @staticmethod
    def _current_hour_start() -> datetime:
        now = datetime.now(timezone.utc)
        return now.replace(minute=0, second=0, microsecond=0)

    @staticmethod
    def _seconds_until_next_hour() -> float:
        now = datetime.now(timezone.utc)
        next_hour = now.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
        return max((next_hour - now).total_seconds() + 1, 1)

    async def run(self) -> None:
        """Entry point — run as a long-lived asyncio task."""
        bt.logging.info("[QueryScheduler] Starting (local synthetic generation)")

        previous_time_range: Optional[datetime] = None
        previous_epoch_dispatched = False
        previous_allocations: dict[str, dict[int, int]] = {}

        while True:
            try:
                time_range_start = self._current_hour_start()

                # Promote pending declared changes before scoring so the just-ended
                # window ramps against current declared capacity.
                if (
                    previous_time_range is not None
                    and time_range_start != previous_time_range
                ):
                    if previous_epoch_dispatched:
                        promoted = await miner_db.promote_pending_declared()

                        if promoted:
                            bt.logging.info(
                                f"[QueryScheduler] Promoted {promoted} "
                                f"pending declared updates"
                            )

                        bt.logging.info(
                            f"[QueryScheduler] Hour boundary: scoring epoch "
                            f"{previous_time_range.isoformat()}"
                        )

                        asyncio.create_task(
                            self.score_epoch(previous_time_range, previous_allocations)
                        )
                    else:
                        bt.logging.info(
                            "[QueryScheduler] Previous epoch had no dispatch "
                            "window — skipping scoring."
                        )
                    previous_epoch_dispatched = False

                previous_time_range = time_range_start

                elapsed_at_start = (
                    datetime.now(timezone.utc) - time_range_start
                ).total_seconds()
                remaining = self.SPREAD_SECONDS - elapsed_at_start

                if remaining < self.MIN_DISPATCH_WINDOW_SECONDS:
                    bt.logging.info(
                        f"[QueryScheduler] Only {remaining:.0f}s remain in epoch "
                        f"(< {self.MIN_DISPATCH_WINDOW_SECONDS}s minimum) — "
                        f"skipping dispatch."
                    )
                    await asyncio.sleep(self._seconds_until_next_hour())
                    continue

                available_uids = list(self.neuron.available_uids)

                if not available_uids:
                    bt.logging.warning(
                        "[QueryScheduler] No available UIDs, waiting for next hour."
                    )
                    await asyncio.sleep(self._seconds_until_next_hour())
                    continue

                allocations_by_type: dict[str, dict[int, int]] = {}
                for st in SEARCH_TYPES:
                    rows = await miner_db.get_allocation_state(st)
                    allocations_by_type[st] = {
                        uid: rows.get(uid, (0.0, 0, DEFAULT_PER_UID))[2]
                        for uid in available_uids
                    }

                previous_allocations = allocations_by_type

                items = await self.generator.generate_epoch_queries(
                    available_uids,
                    verified_by_type=allocations_by_type,
                    scoring_model=self.neuron.config.neuron.scoring_model,
                )

                previous_epoch_dispatched = True

                bt.logging.info(
                    f"[QueryScheduler] {len(items)} queries ready for "
                    f"{time_range_start.isoformat()} "
                    f"across {len(available_uids)} UIDs"
                )

                await self._dispatch_epoch(items, time_range_start)

                sleep_seconds = self._seconds_until_next_hour()
                bt.logging.info(
                    f"[QueryScheduler] All queries dispatched for "
                    f"{time_range_start.isoformat()}. "
                    f"Waiting {sleep_seconds:.1f}s for next UTC hour..."
                )
                await asyncio.sleep(sleep_seconds)

            except Exception as e:
                bt.logging.error(f"[QueryScheduler] Unexpected error: {e}")
                await asyncio.sleep(5)
