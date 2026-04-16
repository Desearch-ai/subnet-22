import asyncio
import time
from datetime import datetime, timedelta, timezone
from typing import Dict, Optional

import bittensor as bt
import torch

from neurons.validators.scoring.scoring_store import ScoringStore
from neurons.validators.scoring.synthetic_query_generator import SyntheticQueryGenerator


class QueryScheduler:
    """
    Background scheduler that drives scoring queries using locally-generated
    synthetic questions.

    Lifecycle per UTC hour:
      1. Batch-generate all queries for every active UID via SyntheticQueryGenerator.
         Epoch-level params (tools, date_filter) are shared; only question text varies.
      2. Dispatch each query at its random fire time spread over ~55 minutes.
      3. Save the miner's response in ScoringStore.
      4. On hour boundary -> score the previous hour's responses.

    Replaces the previous utility-API-polling approach.  Each validator now
    generates its own synthetics independently.
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
        scoring_seed: Optional[int] = None,
    ) -> None:
        """Send one scoring query to a specific miner and persist the response."""
        try:
            validator = self.validators[search_type]
            response = await validator.send_scoring_query(query, uid=uid)
            if response is not None:
                await self.scoring_store.save_response(
                    time_range_start,
                    uid,
                    search_type,
                    response,
                    scoring_seed=scoring_seed,
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
    ) -> None:
        """Score one search type for a completed epoch."""
        validator = self.validators.get(search_type)
        if validator is None or not items:
            return

        uids = torch.tensor([item["uid"] for item in items])
        responses = [item["response"] for item in items]
        scoring_seeds = [item.get("scoring_seed") for item in items]
        prompts = [self._extract_prompt(response) for response in responses]
        event = {}

        bt.logging.info(
            f"[QueryScheduler] Scoring {search_type}: {len(items)} responses"
        )

        await validator.compute_rewards_and_penalties(
            event=event,
            prompts=prompts,
            responses=responses,
            uids=uids,
            start_time=time.time(),
            scoring_epoch_start=time_range_start,
            scoring_seeds=scoring_seeds,
        )

    async def score_epoch(self, time_range_start: datetime) -> None:
        """Load all responses for a completed hour and run reward/penalty computation."""
        try:
            bt.logging.info(
                f"[QueryScheduler] Scoring epoch {time_range_start.isoformat()}"
            )
            all_responses = await self.scoring_store.get_all_for_range(time_range_start)

            if not all_responses:
                bt.logging.warning(
                    f"[QueryScheduler] No responses for epoch "
                    f"{time_range_start.isoformat()}, skipping scoring."
                )
                return

            score_tasks = [
                self._score_search_type(search_type, items, time_range_start)
                for search_type, items in all_responses.items()
                if self.validators.get(search_type) is not None and items
            ]

            if not score_tasks:
                bt.logging.warning(
                    f"[QueryScheduler] No scoreable responses for epoch "
                    f"{time_range_start.isoformat()}, skipping scoring."
                )
                return

            results = await asyncio.gather(*score_tasks, return_exceptions=True)

            for result in results:
                if isinstance(result, Exception):
                    bt.logging.error(
                        f"[QueryScheduler] Error scoring epoch task: {result}"
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

                # Batch-generate all queries for this epoch
                items = await self.generator.generate_epoch_queries(
                    available_uids, self.SPREAD_SECONDS
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
                            scoring_seed=item["scoring_seed"],
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
