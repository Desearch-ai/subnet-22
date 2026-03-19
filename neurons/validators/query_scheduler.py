import asyncio
import time
from datetime import datetime, timedelta, timezone
from typing import Dict, Optional

import bittensor as bt
import torch

from neurons.validators.scoring_store import ScoringStore
from neurons.validators.utility_api_client import UtilityAPIClient


class QueryScheduler:
    """
    Background scheduler that drives scoring queries from the utility API.

    Lifecycle per UTC hour:
      1. Poll utility API for (uid, search_type, question) items.
      2. Dispatch each as a fire-and-forget scoring query to the target miner.
      3. Save the miner's response in ScoringStore.
      4. On hour boundary → score the previous hour's responses (if not first epoch).

    The API returns 404 when all questions for the current hour are served;
    we sleep until the next UTC hour begins and then resume polling.
    """

    def __init__(
        self,
        neuron,
        utility_api: UtilityAPIClient,
        scoring_store: ScoringStore,
        validators: Dict,  # {"ai_search": ..., "x_search": ..., "web_search": ...}
    ):
        self.neuron = neuron
        self.utility_api = utility_api
        self.scoring_store = scoring_store
        self.validators = validators

        self.current_time_range: Optional[datetime] = None
        self.min_request_interval_seconds = 4.1
        self._next_poll_at = 0.0

        # Skip scoring on the first epoch boundary (incomplete responses)
        self.is_first_epoch = True

    def _build_query(self, question_query: str, params: dict) -> dict:
        """Format a query dict for the appropriate validator's send_scoring_query()."""
        return {"query": question_query, **params}

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
                await self.scoring_store.save_response(
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
    ) -> None:
        """Score one search type for a completed epoch."""
        validator = self.validators.get(search_type)
        if validator is None or not items:
            return

        uids = torch.tensor([item["uid"] for item in items])
        responses = [item["response"] for item in items]
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

    async def _wait_for_poll_window(self) -> None:
        delay = self._next_poll_at - time.monotonic()
        if delay > 0:
            await asyncio.sleep(delay)

    def _schedule_next_poll(self, request_started_at: float) -> None:
        self._next_poll_at = request_started_at + self.min_request_interval_seconds

    def _seconds_until_next_hour(self) -> float:
        now = datetime.now(timezone.utc)
        next_hour = now.replace(minute=0, second=0, microsecond=0) + timedelta(
            hours=1
        )
        return max((next_hour - now).total_seconds() + 1, 1)

    async def run(self) -> None:
        """Entry point — run as a long-lived asyncio task."""
        bt.logging.info("[QueryScheduler] Starting")

        while True:
            request_started_at: Optional[float] = None
            try:
                await self._wait_for_poll_window()
                request_started_at = time.monotonic()
                item = await self.utility_api.fetch_next_question()
                self._schedule_next_poll(request_started_at)

                time_range_start = item["time_range_start"]
                if isinstance(time_range_start, str):
                    time_range_start = datetime.fromisoformat(
                        time_range_start.replace("Z", "+00:00")
                    )

                uid: int = item["uid"]
                search_type: str = item["search_type"]
                question_query: str = item["question"]["query"]
                params: dict = item["question"].get("params", {})

                # Detect hour boundary
                if (
                    self.current_time_range is not None
                    and time_range_start != self.current_time_range
                ):
                    bt.logging.info(
                        "[QueryScheduler] Hour boundary detected "
                        f"previous={self.current_time_range.isoformat()} "
                        f"next={time_range_start.isoformat()} "
                        f"is_first_epoch={self.is_first_epoch}"
                    )
                    if not self.is_first_epoch:
                        asyncio.create_task(self.score_epoch(self.current_time_range))
                    else:
                        bt.logging.info(
                            "[QueryScheduler] First epoch boundary — "
                            "skipping scoring (incomplete data)."
                        )
                    self.is_first_epoch = False

                self.current_time_range = time_range_start

                # Dispatch scoring query in background
                query = self._build_query(question_query, params)

                asyncio.create_task(
                    self._send_and_save(search_type, uid, query, time_range_start)
                )

            except Exception as e:
                status = getattr(e, "status", None)

                if status == 404:
                    # All questions for this hour have been served
                    sleep_seconds = self._seconds_until_next_hour()
                    self._next_poll_at = 0.0
                    bt.logging.info(
                        "[QueryScheduler] All questions served for current hour. "
                        f"Waiting {sleep_seconds:.1f}s for next UTC hour..."
                    )
                    await asyncio.sleep(sleep_seconds)
                elif status == 429:
                    # Unexpected rate limit — re-align to the server window.
                    if request_started_at is not None:
                        self._schedule_next_poll(request_started_at)
                    else:
                        self._next_poll_at = time.monotonic() + (
                            self.min_request_interval_seconds
                        )
                    await self._wait_for_poll_window()
                else:
                    bt.logging.error(
                        "[QueryScheduler] Unexpected error while polling "
                        f"dataset/next current_time_range="
                        f"{self.current_time_range.isoformat() if self.current_time_range else None} "
                        f"status={status}: {e}"
                    )
                    await asyncio.sleep(5)
