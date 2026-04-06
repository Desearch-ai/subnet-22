import asyncio
import time
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Dict, Optional

import bittensor as bt
import torch

from neurons.validators.scoring_dataset import (
    SCORING_CONCURRENCY,
    ScoringAssignment,
    build_question_pool,
    build_scoring_assignments,
    current_scoring_window,
)
from neurons.validators.seed_commitment import build_window_seed_state

if TYPE_CHECKING:
    from neurons.validators.scoring_store import ScoringStore


class QueryScheduler:
    """
    Background scheduler that drives scoring queries from a local Hugging Face dataset.

    Lifecycle per UTC hour:
      1. Load or reuse the local dataset question pool.
      2. Publish this validator's seed commitment.
      3. After the reveal delay, fetch current validator commitments and derive a
         shared combined seed.
      4. Build deterministic (uid, search_type, question) assignments locally.
      5. Execute one ai/x/web scoring query for every miner and save the
         responses in ScoringStore.
      6. On the next hour boundary, score the previous hour's saved responses.

    The Utility API remains available for log submission and backward compatibility,
    but scheduling no longer depends on `/dataset/next`.
    """

    def __init__(
        self,
        neuron,
        scoring_store: "ScoringStore",
        validators: Dict,  # {"ai_search": ..., "x_search": ..., "web_search": ...}
    ):
        self.neuron = neuron
        self.scoring_store = scoring_store
        self.validators = validators
        self.question_pool = build_question_pool()
        self.scoring_concurrency = SCORING_CONCURRENCY

        self.current_time_range: Optional[datetime] = None

        # Skip scoring on the first epoch boundary (incomplete responses)
        self.is_first_epoch = True

    def _build_query(self, question_query: str, params: dict) -> dict:
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

    async def _send_and_save(self, assignment: ScoringAssignment) -> None:
        try:
            validator = self.validators[assignment.search_type]
            query = self._build_query(
                assignment.question.query,
                assignment.question.params,
            )
            response = await validator.send_scoring_query(query, uid=assignment.uid)
            if response is not None:
                await self.scoring_store.save_response(
                    assignment.time_range_start,
                    assignment.uid,
                    assignment.search_type,
                    response,
                    scoring_seed=assignment.scoring_seed,
                )
                bt.logging.debug(
                    "[QueryScheduler] Saved response "
                    f"uid={assignment.uid} type={assignment.search_type}"
                )
        except Exception as e:
            bt.logging.error(
                "[QueryScheduler] Scoring query failed "
                f"uid={assignment.uid} type={assignment.search_type}: {e}"
            )

    async def _run_assignment(
        self,
        assignment: ScoringAssignment,
        semaphore: asyncio.Semaphore,
    ) -> None:
        async with semaphore:
            await self._send_and_save(assignment)

    async def _score_search_type(
        self,
        search_type: str,
        items: list,
        time_range_start: datetime,
    ) -> None:
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
        try:
            bt.logging.info(
                f"[QueryScheduler] Scoring epoch {time_range_start.isoformat()}"
            )
            all_responses = await self.scoring_store.get_all_for_range(time_range_start)

            if not all_responses:
                bt.logging.warning(
                    "[QueryScheduler] No responses for epoch "
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
                    "[QueryScheduler] No scoreable responses for epoch "
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

    def _collect_miner_uids(self) -> list[int]:
        metagraph_uids = self.neuron.metagraph.uids
        if hasattr(metagraph_uids, "tolist"):
            return [int(uid) for uid in metagraph_uids.tolist()]

        return [int(uid) for uid in metagraph_uids]

    async def _run_window(self, time_range_start: datetime) -> None:
        validators = await self.neuron.get_validators()

        window_state = await build_window_seed_state(
            subtensor=self.neuron.subtensor,
            wallet=self.neuron.wallet,
            netuid=self.neuron.config.netuid,
            uid=self.neuron.uid,
            validators=validators,
            time_range_start=time_range_start,
        )

        if window_state.validator_count == 0:
            bt.logging.warning(
                "[QueryScheduler] No validator commitments available for "
                f"{time_range_start.isoformat()}, skipping window."
            )
            return

        miner_uids = self._collect_miner_uids()

        assignments = build_scoring_assignments(
            time_range_start=time_range_start,
            miner_uids=miner_uids,
            question_pool=self.question_pool,
            combined_seed=window_state.combined_seed,
        )

        bt.logging.info(
            "[QueryScheduler] Built local scoring plan "
            f"time_range={time_range_start.isoformat()} "
            f"validators={window_state.validator_count} "
            f"assignments_total={len(assignments)} "
            f"combined_seed={window_state.combined_seed}"
        )

        if not assignments:
            return

        semaphore = asyncio.Semaphore(self.scoring_concurrency)

        tasks = [
            asyncio.create_task(self._run_assignment(assignment, semaphore))
            for assignment in assignments
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        for result in results:
            if isinstance(result, Exception):
                bt.logging.error(f"[QueryScheduler] Window task failed: {result}")

    async def _sleep_until_next_window(self, time_range_start: datetime) -> None:
        next_window_start = time_range_start + timedelta(hours=1)
        delay = (next_window_start - datetime.now(timezone.utc)).total_seconds()
        if delay > 0:
            await asyncio.sleep(delay)

    async def run(self) -> None:
        bt.logging.info("[QueryScheduler] Starting")
        await self.question_pool.initialize()

        while True:
            try:
                window_start = current_scoring_window()

                if (
                    self.current_time_range is not None
                    and window_start != self.current_time_range
                ):
                    bt.logging.info(
                        "[QueryScheduler] Hour boundary detected "
                        f"previous={self.current_time_range.isoformat()} "
                        f"next={window_start.isoformat()} "
                        f"is_first_epoch={self.is_first_epoch}"
                    )

                    if not self.is_first_epoch:
                        await self.score_epoch(self.current_time_range)
                    else:
                        bt.logging.info(
                            "[QueryScheduler] First epoch boundary - "
                            "skipping scoring (incomplete data)."
                        )

                    self.is_first_epoch = False

                self.current_time_range = window_start
                await self._run_window(window_start)
                await self._sleep_until_next_window(window_start)
            except Exception as e:
                bt.logging.error(
                    "[QueryScheduler] Unexpected scheduler error "
                    f"time_range={self.current_time_range.isoformat() if self.current_time_range else None}: {e}"
                )
                await asyncio.sleep(5)
