import asyncio
import random
import time
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Dict, Optional

import bittensor as bt
import torch

from neurons.validators.scoring import capacity
from neurons.validators.scoring.scoring_store import SEARCH_TYPES, ScoringStore
from neurons.validators.scoring.synthetic_query_generator import SyntheticQueryGenerator

SEARCH_TYPE_WEIGHTS = {
    "ai_search": 0.50,
    "x_search": 0.25,
    "web_search": 0.25,
}

ORGANIC_DEEP_SCORE_WEIGHT = 5
ORGANIC_SCORE_CAP_PER_TYPE = 100


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

        # Skip scoring on the first epoch boundary (incomplete responses)
        self.is_first_epoch = True

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

    async def _score_search_type(
        self,
        search_type: str,
        items: list,
        time_range_start: datetime,
    ) -> dict[int, float]:
        """
        Score one search type for a completed epoch.

        Each item carries a ``kind`` of "synthetic" or "organic". Organic rewards
        count ``ORGANIC_DEEP_SCORE_WEIGHT`` times when computing the per-UID
        mean quality.
        """
        validator = self.validators.get(search_type)
        if validator is None or not items:
            return {}

        uids = torch.tensor([item["uid"] for item in items])
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

        for uid_tensor, reward, kind in zip(uids, rewards.tolist(), kinds):
            uid = uid_tensor.item()
            weight = ORGANIC_DEEP_SCORE_WEIGHT if kind == "organic" else 1
            uid_totals[uid] += weight * reward
            uid_weights[uid] += weight

        return {uid: uid_totals[uid] / uid_weights[uid] for uid in uid_totals}

    def _sample_organics(self, organics: list) -> list:
        """Cap organics per search type using uniform random sampling."""
        if len(organics) <= ORGANIC_SCORE_CAP_PER_TYPE:
            return organics
        return random.sample(organics, ORGANIC_SCORE_CAP_PER_TYPE)

    async def score_epoch(self, time_range_start: datetime) -> None:
        """Load all responses for a completed hour and run reward/penalty computation."""
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

            qualities_per_type: dict[str, dict[int, float]] = {}

            for search_type in SEARCH_TYPES:
                if self.validators.get(search_type) is None:
                    continue

                synth_items = [
                    {**item, "kind": "synthetic"}
                    for item in synthetics.get(search_type, [])
                ]
                organic_pool = organics.get(search_type, [])
                organic_sample = self._sample_organics(organic_pool)
                organic_items = [{**item, "kind": "organic"} for item in organic_sample]

                merged = synth_items + organic_items

                if not merged:
                    continue

                bt.logging.info(
                    f"[QueryScheduler] {search_type}: "
                    f"{len(synth_items)} synthetic + {len(organic_items)} organic "
                    f"(pool={len(organic_pool)}, cap={ORGANIC_SCORE_CAP_PER_TYPE})"
                )

                try:
                    uid_qualities = await self._score_search_type(
                        search_type, merged, time_range_start
                    )
                    qualities_per_type[search_type] = uid_qualities

                    for uid, quality in uid_qualities.items():
                        await capacity.update_after_scoring(
                            uid=uid,
                            search_type=search_type,
                            quality=quality,
                            window_start=window_start,
                        )
                except Exception as e:
                    bt.logging.error(
                        f"[QueryScheduler] Error scoring {search_type}: {e}"
                    )

            # Combine per-type scores into one weighted score per UID
            # AI 50%, X 25%, Web 25% — then one EMA update
            all_uids: set[int] = set()
            for uid_q in qualities_per_type.values():
                all_uids.update(uid_q.keys())

            if all_uids:
                combined: dict[int, float] = {}
                for uid in all_uids:
                    score = 0.0
                    for st, weight in SEARCH_TYPE_WEIGHTS.items():
                        score += weight * qualities_per_type.get(st, {}).get(uid, 0.0)
                    combined[uid] = score

                uids_tensor = torch.tensor(
                    list(combined.keys()),
                    dtype=torch.long,
                    device=self.neuron.config.neuron.device,
                )
                rewards_tensor = torch.tensor(
                    list(combined.values()),
                    dtype=torch.float32,
                    device=self.neuron.config.neuron.device,
                )
                await self.neuron.update_moving_averaged_scores(
                    uids_tensor, rewards_tensor
                )

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

        while True:
            try:
                time_range_start = self._current_hour_start()

                # Score the previous hour on boundary crossing
                if (
                    previous_time_range is not None
                    and time_range_start != previous_time_range
                ):
                    if not self.is_first_epoch:
                        bt.logging.info(
                            f"[QueryScheduler] Hour boundary: scoring epoch "
                            f"{previous_time_range.isoformat()}"
                        )
                        asyncio.create_task(self.score_epoch(previous_time_range))
                    else:
                        bt.logging.info(
                            "[QueryScheduler] First epoch boundary — "
                            "skipping scoring (incomplete data)."
                        )
                    self.is_first_epoch = False

                previous_time_range = time_range_start

                # Snapshot the currently available UIDs
                available_uids = list(self.neuron.available_uids)
                if not available_uids:
                    bt.logging.warning(
                        "[QueryScheduler] No available UIDs, waiting for next hour."
                    )
                    await asyncio.sleep(self._seconds_until_next_hour())
                    continue

                # Fetch verified concurrency for all UIDs
                verified_by_type: dict[str, dict[int, int]] = {}
                for st in SEARCH_TYPES:
                    verified_by_type[st] = await capacity.get_all_verified(st)

                # Batch-generate all queries for this epoch
                items = await self.generator.generate_epoch_queries(
                    available_uids,
                    self.SPREAD_SECONDS,
                    verified_by_type=verified_by_type,
                )
                bt.logging.info(
                    f"[QueryScheduler] {len(items)} queries ready for "
                    f"{time_range_start.isoformat()} "
                    f"across {len(available_uids)} UIDs"
                )

                # Dispatch each pre-generated item at its scheduled fire time
                for item in items:
                    # Abort if the hour has changed (new epoch)
                    current_hour = self._current_hour_start()
                    if current_hour != time_range_start:
                        bt.logging.info(
                            "[QueryScheduler] Hour changed during dispatch, "
                            "breaking to start new epoch."
                        )
                        break

                    # Seconds elapsed since this hour started
                    elapsed = (
                        datetime.now(timezone.utc) - time_range_start
                    ).total_seconds()

                    wait_seconds = item["delay_seconds"] - elapsed
                    if wait_seconds > 0:
                        await asyncio.sleep(wait_seconds)

                    # Fire-and-forget dispatch
                    asyncio.create_task(
                        self._send_and_save(
                            item["search_type"],
                            item["uid"],
                            item["query"],
                            time_range_start,
                        )
                    )

                # All items dispatched (or hour changed) — wait for next hour
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
