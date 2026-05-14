import asyncio
import random
import time
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Dict, Optional

import bittensor as bt
import numpy as np

from neurons.validators.scoring import capacity, miner_db
from neurons.validators.scoring.scoring_store import SEARCH_TYPES, ScoringStore
from neurons.validators.scoring.synthetic_query_generator import SyntheticQueryGenerator

SEARCH_TYPE_WEIGHTS = {
    "ai_search": 0.60,
    "x_search": 0.20,
    "web_search": 0.20,
}

ORGANIC_DEEP_SCORE_WEIGHT = 5
ORGANIC_SCORE_CAP_PER_TYPE = 100

# Superlinear incentive formula: score = quality^alpha * volume^beta.
# Both >1 make consolidating a UID strictly more profitable than splitting,
# so two half-volume sybil UIDs always earn less than one full-volume UID.
QUALITY_EXPONENT_ALPHA = 1.5
VOLUME_EXPONENT_BETA = 1.5

BATCH_SIZE = 20
BATCH_INTERVAL_SECONDS = 5
GROUP_SIZE = 5


def combine_superlinear_scores(
    qualities_per_type: dict[str, dict[int, tuple[float, int]]],
) -> dict[int, float]:
    """
    Fold per-type ``(quality, volume)`` into one superlinear score per UID:
    ``score = (sum_t w_t * q_t)^alpha * (sum_t w_t * v_t)^beta`` where
    ``w_t`` is ``SEARCH_TYPE_WEIGHTS[t]``.
    """
    all_uids: set[int] = set()
    for uid_q in qualities_per_type.values():
        all_uids.update(uid_q.keys())

    combined: dict[int, float] = {}
    for uid in all_uids:
        q_combined = 0.0
        v_combined = 0.0
        for st, weight in SEARCH_TYPE_WEIGHTS.items():
            q, v = qualities_per_type.get(st, {}).get(uid, (0.0, 0))
            q_combined += weight * q
            v_combined += weight * v
        combined[uid] = (
            q_combined**QUALITY_EXPONENT_ALPHA * v_combined**VOLUME_EXPONENT_BETA
        )
    return combined


class QueryScheduler:
    """
    Background scheduler that drives scoring queries using locally-generated
    synthetic questions.

    Lifecycle per UTC hour:
      1. Batch-generate all queries for every active UID via SyntheticQueryGenerator.
         Epoch-level params (tools, date_filter) are shared; only question text varies.
         Each miner gets N queries per search type where N = verified concurrency.
      2. Dispatch each query at its random fire time spread over ~55 minutes.
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
        """Dispatch AI, X, Web sequentially. Each phase walks sorted UIDs in
        groups of GROUP_SIZE; UIDs within a group run concurrently."""
        for search_type in SEARCH_TYPES:
            if self._current_hour_start() != time_range_start:
                return

            grouped: dict[int, list] = defaultdict(list)
            for item in items:
                if item["search_type"] == search_type:
                    grouped[item["uid"]].append(item)

            sorted_uids = sorted(grouped)
            bt.logging.info(
                f"[QueryScheduler] Phase {search_type}: "
                f"{sum(len(v) for v in grouped.values())} queries "
                f"across {len(sorted_uids)} UIDs in groups of {GROUP_SIZE}"
            )

            for start in range(0, len(sorted_uids), GROUP_SIZE):
                if self._current_hour_start() != time_range_start:
                    return
                group = sorted_uids[start : start + GROUP_SIZE]
                await asyncio.gather(
                    *[
                        self._dispatch_uid(
                            uid, search_type, grouped[uid], time_range_start
                        )
                        for uid in group
                    ],
                    return_exceptions=True,
                )

    async def _dispatch_uid(
        self,
        uid: int,
        search_type: str,
        uid_items: list,
        time_range_start: datetime,
    ) -> None:
        """Fire one UID's queries in BATCH_SIZE bursts, BATCH_INTERVAL_SECONDS apart."""
        batches = [
            uid_items[i : i + BATCH_SIZE]
            for i in range(0, len(uid_items), BATCH_SIZE)
        ]

        for batch_idx, batch in enumerate(batches):
            if self._current_hour_start() != time_range_start:
                return

            if batch_idx > 0:
                await asyncio.sleep(BATCH_INTERVAL_SECONDS)

            # Fire-and-forget so the next batch fires on schedule regardless of miner latency.
            for item in batch:
                asyncio.create_task(
                    self._send_and_save(
                        search_type, uid, item["query"], time_range_start
                    )
                )

    async def _score_search_type(
        self,
        search_type: str,
        items: list,
        time_range_start: datetime,
    ) -> dict[int, tuple[float, int]]:
        """
        Score one search type for a completed epoch. Returns per-UID
        ``(quality, volume)``: quality is a 5x-organic-weighted mean of
        per-response rewards (0-1); volume is the raw count of responses
        scored for that UID.
        """
        validator = self.validators.get(search_type)
        if validator is None or not items:
            return {}

        uids = np.array([item["uid"] for item in items], dtype=np.int64)
        responses = [item["response"] for item in items]
        kinds = [item.get("kind", "synthetic") for item in items]
        prompts = [self._extract_prompt(response) for response in responses]
        event = {}

        bt.logging.info(
            f"[QueryScheduler] Scoring {search_type}: {len(items)} responses"
        )

        result = await validator.compute_rewards_and_penalties(
            event=event,
            prompts=prompts,
            responses=responses,
            uids=uids,
            start_time=time.time(),
            scoring_epoch_start=time_range_start,
        )

        if result is None:
            return {}

        rewards = result[0]

        uid_totals: dict[int, float] = defaultdict(float)
        uid_weights: dict[int, float] = defaultdict(float)
        uid_volumes: dict[int, int] = defaultdict(int)

        for uid_tensor, reward, kind in zip(uids, rewards.tolist(), kinds):
            uid = uid_tensor.item()
            weight = ORGANIC_DEEP_SCORE_WEIGHT if kind == "organic" else 1
            uid_totals[uid] += weight * reward
            uid_weights[uid] += weight
            uid_volumes[uid] += 1

        return {
            uid: (uid_totals[uid] / uid_weights[uid], uid_volumes[uid])
            for uid in uid_totals
        }

    def _sample_organics(self, organics: list) -> list:
        """Cap organics per search type using uniform random sampling."""
        if len(organics) <= ORGANIC_SCORE_CAP_PER_TYPE:
            return organics
        return random.sample(organics, ORGANIC_SCORE_CAP_PER_TYPE)

    async def _score_one_type(
        self,
        search_type: str,
        synthetics: dict,
        organics: dict,
        time_range_start: datetime,
        window_start: str,
        allocations: dict[int, int],
    ) -> dict[int, tuple[float, int]]:
        """Merge synth + organic sample for one type, score it, and update
        capacity per UID. Returns per-UID ``(quality, volume)`` for the combine
        step, or an empty dict if nothing scored."""
        if self.validators.get(search_type) is None:
            return {}

        synth_items = [
            {**item, "kind": "synthetic"} for item in synthetics.get(search_type, [])
        ]
        organic_pool = organics.get(search_type, [])
        organic_sample = self._sample_organics(organic_pool)
        organic_items = [{**item, "kind": "organic"} for item in organic_sample]

        merged = synth_items + organic_items
        if not merged:
            return {}

        bt.logging.info(
            f"[QueryScheduler] {search_type}: "
            f"{len(synth_items)} synthetic + {len(organic_items)} organic "
            f"(pool={len(organic_pool)}, cap={ORGANIC_SCORE_CAP_PER_TYPE})"
        )

        try:
            uid_results = await self._score_search_type(
                search_type, merged, time_range_start
            )
        except Exception as e:
            bt.logging.error(f"[QueryScheduler] Error scoring {search_type}: {e}")
            return {}

        for uid, (quality, _volume) in uid_results.items():
            await capacity.update_after_scoring(
                uid=uid,
                search_type=search_type,
                quality=quality,
                window_start=window_start,
                allocated=allocations.get(uid, capacity.DEFAULT_PER_UID),
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

                # On hour boundary: promote any staged declared-concurrency
                # changes *before* firing scoring, so the ramp for the window
                # we just ended sees the miner's current declared capability.
                # Scoring then runs in the background so this hour's dispatch
                # keeps its full 55-minute spread. Pass the previous epoch's
                # allocations so historical window rows record the budget
                # that was actually active, not the one we're about to write.
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
                            self.score_epoch(
                                previous_time_range, previous_allocations
                            )
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

                # Snapshot the currently available UIDs
                available_uids = list(self.neuron.available_uids)

                if not available_uids:
                    bt.logging.warning(
                        "[QueryScheduler] No available UIDs, waiting for next hour."
                    )
                    await asyncio.sleep(self._seconds_until_next_hour())
                    continue

                # Allocate the per-type budget across active UIDs by quality
                allocations_by_type: dict[str, dict[int, int]] = {}

                for st in SEARCH_TYPES:
                    rows = await miner_db.get_allocation_state(st)
                    default = (0.0, capacity.DEFAULT_PER_UID, capacity.DEFAULT_PER_UID)
                    quality_for_active = {
                        uid: rows.get(uid, default)[0] for uid in available_uids
                    }
                    declared_for_active = {
                        uid: rows.get(uid, default)[1] for uid in available_uids
                    }
                    prev_alloc_for_active = {
                        uid: rows.get(uid, default)[2] for uid in available_uids
                    }
                    allocations_by_type[st] = capacity.allocate_synthetic_budget(
                        quality_for_active,
                        declared_for_active,
                        prev_alloc_for_active,
                    )
                    await miner_db.bulk_update_verified(st, allocations_by_type[st])

                previous_allocations = allocations_by_type

                items = await self.generator.generate_epoch_queries(
                    available_uids,
                    verified_by_type=allocations_by_type,
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
