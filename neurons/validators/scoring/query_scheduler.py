import asyncio
import math
import random
import time
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Dict, Optional

import bittensor as bt
import numpy as np

from desearch.miner_config import LANES, SearchType, lane_key
from desearch.protocol import SearchMode
from neurons.validators.scoring import capacity, miner_db
from neurons.validators.scoring.constants import (
    DEFAULT_PER_UID,
    GATE_RAMP,
    MIN_DEEP_SAMPLES_PER_POOL,
    POOL_SHARES,
    QUALITY_EXPONENT,
    QUALITY_THRESHOLDS,
    VOLUME_EXPONENT,
)
from neurons.validators.scoring.scoring_store import SEARCH_TYPES, ScoringStore
from neurons.validators.scoring.synthetic_query_generator import (
    SyntheticQueryGenerator,
    _weighted_counts,
)

ORGANIC_VALUE_MULTIPLIER = 3
ORGANIC_DEEP_CAP_PER_TYPE = 100

BATCH_FRACTION = 0.50
BATCH_INTERVAL_SECONDS = 3
GROUP_SIZE = 2

DEEP_SAMPLE_RATE = 0.20
DEEP_SAMPLE_FLOOR = 3
DEEP_SAMPLE_WEIGHT = 5


def _unpack_quality(val: tuple) -> tuple[float, float, float]:
    if len(val) >= 3:
        return float(val[0]), float(val[1]), float(val[2])
    q, v = val
    return float(q), float(q), float(v)


def gate_for(q_gate: float, search_type: SearchType) -> float:
    thr = QUALITY_THRESHOLDS[search_type]
    return min(1.0, max(0.0, (q_gate - (thr - GATE_RAMP)) / GATE_RAMP))


def _pool_raw_scores(
    search_type: SearchType, uid_results: dict[int, tuple]
) -> dict[int, float]:
    raw: dict[int, float] = {}
    for uid, (q_gate, q_weight, volume, samples) in uid_results.items():
        if volume <= 0 or samples < MIN_DEEP_SAMPLES_PER_POOL:
            continue
        gate = gate_for(q_gate, search_type)
        if gate <= 0:
            continue
        raw[uid] = gate * q_weight**QUALITY_EXPONENT * volume**VOLUME_EXPONENT
    return raw


def combine_pool_scores(
    qualities_per_pool: dict[tuple[SearchType, Optional[SearchMode]], dict[int, tuple]],
) -> dict[int, float]:
    scores: dict[int, float] = defaultdict(float)

    for pool, share in POOL_SHARES.items():
        uid_results = qualities_per_pool.get(pool) or {}
        if not uid_results:
            continue

        raw = _pool_raw_scores(pool[0], uid_results)
        total = sum(raw.values())
        if total <= 0:
            continue

        for uid, uid_raw in raw.items():
            scores[uid] += share * uid_raw / total

    return dict(scores)


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
        validators: Dict,  # {"ai_search": ..., "x_search": ...}
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
            uid_items[i : i + batch_size] for i in range(0, len(uid_items), batch_size)
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
        by_uid: dict[int, list[int]] = defaultdict(list)
        for idx, item in enumerate(synth_items):
            by_uid[item["uid"]].append(idx)
        sampled: set[int] = set()
        for indices in by_uid.values():
            n = min(
                len(indices),
                max(DEEP_SAMPLE_FLOOR, round(len(indices) * DEEP_SAMPLE_RATE)),
            )
            sampled.update(self._proportional_pick(indices, synth_items, n))
        return sampled

    @staticmethod
    def _deep_combo_key(item):
        resp = item.get("response")
        mode = getattr(resp, "mode", None)
        result_type = getattr(resp, "result_type", None)
        tools = getattr(resp, "tools", None)
        return (
            getattr(mode, "value", mode),
            getattr(result_type, "value", result_type),
            tuple(tools) if tools else (),
        )

    def _proportional_pick(self, indices: list, synth_items: list, n: int) -> list:
        buckets: dict = defaultdict(list)
        for idx in indices:
            buckets[self._deep_combo_key(synth_items[idx])].append(idx)
        members = list(buckets.values())
        total = len(indices)
        counts = _weighted_counts(n, [len(m) / total for m in members])
        picked: list = []
        for m, c in zip(members, counts):
            random.shuffle(m)
            picked.extend(m[: min(c, len(m))])
        return picked

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
    ) -> tuple[np.ndarray, np.ndarray]:
        if not items:
            return np.zeros(0, dtype=np.float32), np.zeros(0, dtype=np.float32)
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
            zeros = np.zeros(len(items), dtype=np.float32)
            return zeros, zeros.copy()
        weight_scores = np.asarray(result[0], dtype=np.float32)
        gate_scores = (
            np.asarray(result[5], dtype=np.float32)
            if len(result) > 5
            else weight_scores
        )
        return weight_scores, gate_scores

    async def _score_one_type(
        self,
        search_type: str,
        synthetics: dict,
        organics: dict,
        time_range_start: datetime,
        window_start: str,
        allocations_by_lane: dict[str, dict[int, int]],
    ) -> dict[int, tuple[float, float, int]]:
        """Score synth + organic for one type and update capacity per UID.

        Quality = deep-only weighted mean (synth-deep=DEEP_SAMPLE_WEIGHT,
        organic-deep=ORGANIC_VALUE_MULTIPLIER*DEEP_SAMPLE_WEIGHT) multiplied by
        the per-UID cheap penalty mean in [0, 1]. Cheap contributes no positive
        reward; a UID with no deep items scores 0.

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

        deep_totals: dict[tuple, float] = defaultdict(float)
        gate_totals: dict[tuple, float] = defaultdict(float)
        deep_weights: dict[tuple, float] = defaultdict(float)
        deep_counts: dict[tuple, int] = defaultdict(int)
        cheap_sum: dict[tuple, float] = defaultdict(float)
        cheap_count: dict[tuple, int] = defaultdict(int)
        volumes: dict[tuple, int] = defaultdict(int)

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
                cheap_scores = np.ones(len(cheap_items), dtype=np.float32)
            penalties = cheap_scores.tolist()
            for i, item in enumerate(cheap_items):
                key = (item["uid"], self._item_mode(item))
                cheap_sum[key] += penalties[i]
                cheap_count[key] += 1
                volumes[key] += 1

        deep_items = deep_synth + deep_organic
        if deep_items:
            try:
                full_scores, gate_scores = await self._run_full_scoring(
                    validator, deep_items, time_range_start
                )
            except Exception as e:
                bt.logging.error(
                    f"[QueryScheduler] Full scoring failed {search_type}: {e}"
                )
                full_scores = np.zeros(len(deep_items), dtype=np.float32)
                gate_scores = np.zeros(len(deep_items), dtype=np.float32)
            scores = full_scores.tolist()
            gates = gate_scores.tolist()
            organic_deep_weight = ORGANIC_VALUE_MULTIPLIER * DEEP_SAMPLE_WEIGHT
            for i, item in enumerate(deep_items):
                weight = (
                    DEEP_SAMPLE_WEIGHT if i < len(deep_synth) else organic_deep_weight
                )
                key = (item["uid"], self._item_mode(item))
                deep_totals[key] += weight * scores[i]
                gate_totals[key] += weight * gates[i]
                deep_weights[key] += weight
                deep_counts[key] += 1
                volumes[key] += 1

        results_by_mode: dict[Optional[SearchMode], dict[int, tuple]] = defaultdict(
            dict
        )
        for key in volumes:
            uid, mode = key
            c_uid = cheap_sum[key] / cheap_count[key] if cheap_count[key] > 0 else 1.0
            denom = deep_weights[key]
            q_weight = deep_totals[key] / denom if denom > 0 else 0.0
            q_gate = gate_totals[key] / denom if denom > 0 else 0.0
            results_by_mode[mode][uid] = (
                q_gate * c_uid,
                q_weight * c_uid,
                volumes[key],
                deep_counts[key],
            )

        await self._record_quality(
            search_type, results_by_mode, window_start, allocations_by_lane
        )
        return results_by_mode

    @staticmethod
    def _item_mode(item) -> Optional[SearchMode]:
        mode = getattr(item.get("response"), "mode", None)
        return SearchMode(mode) if mode else None

    async def _record_quality(
        self,
        search_type: SearchType,
        results_by_mode: dict,
        window_start: str,
        allocations_by_lane: dict[str, dict[int, int]],
    ) -> None:
        for mode, uid_results in results_by_mode.items():
            key = lane_key((search_type, mode))
            allocations = allocations_by_lane.get(key, {})
            for uid, (q_gate, _q_weight, _volume, _samples) in uid_results.items():
                await capacity.record_window_quality(
                    uid=uid,
                    search_type=key,
                    quality=q_gate,
                    window_start=window_start,
                    allocated=allocations.get(uid, DEFAULT_PER_UID),
                )

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
            qualities_per_pool: dict[
                tuple[SearchType, Optional[SearchMode]], dict[int, tuple]
            ] = {}

            for search_type in SEARCH_TYPES:
                results_by_mode = await self._score_one_type(
                    search_type,
                    synthetics,
                    organics,
                    time_range_start,
                    window_start,
                    allocations_by_type,
                )
                for mode, uid_results in results_by_mode.items():
                    pool = (search_type, mode)
                    if uid_results and pool in POOL_SHARES:
                        qualities_per_pool[pool] = uid_results

            touched_uids = sorted(
                {uid for results in qualities_per_pool.values() for uid in results}
            )
            await capacity.ramp_after_epoch(touched_uids)

            combined = combine_pool_scores(qualities_per_pool)
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
                for lane in LANES:
                    key = lane_key(lane)
                    rows = await miner_db.get_allocation_state(key)
                    allocations_by_type[key] = {
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
