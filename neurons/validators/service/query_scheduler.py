import asyncio
import time
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Dict, Optional

import bittensor as bt
import torch

from neurons.validators.service.scoring_dataset import (
    SCORING_CONCURRENCY,
    ScoringAssignment,
    build_question_pool,
    build_scoring_assignments,
    current_scoring_window,
    filter_scoring_assignments,
)
from neurons.validators.service.seed_commitment import (
    DEFAULT_BUCKET_COMMITMENT_REVEAL_DELAY_SECONDS,
    WindowBucketState,
    WindowSeedState,
    build_window_bucket_state,
    build_window_seed_state,
    load_window_bucket_state,
    load_window_seed_state,
)

if TYPE_CHECKING:
    from neurons.validators.service.scoring_store import ScoringStore

SCORING_STORAGE_PUBLISH_OFFSET = timedelta(minutes=30)


class QueryScheduler:
    """
    Background scheduler that drives scoring queries from a local Hugging Face dataset.

    Lifecycle per UTC hour:
      1. Load or reuse the local dataset question pool.
      2. Publish this validator's seed commitment.
      3. After the reveal delay, fetch current validator commitments and derive a
         shared combined seed.
      4. Split miner UIDs deterministically across committed validators.
      5. Between `HH:00` and `HH:30`, execute this validator's owned ai/x/web
         scoring queries and save the responses in public object storage.
      6. At `HH:30`, publish this validator's bucket locator, wait for bucket
         commitments, then read the expected stored responses and score them.

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
        self.search_validators = validators
        self.question_pool = build_question_pool()
        self.scoring_concurrency = SCORING_CONCURRENCY

        self.current_time_range: Optional[datetime] = None
        self.window_assignments: dict[datetime, tuple[ScoringAssignment, ...]] = {}
        self.window_bucket_states: dict[datetime, WindowBucketState] = {}

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
            validator = self.search_validators[assignment.search_type]
            query = self._build_query(
                assignment.question.query,
                assignment.question.params,
            )
            response = await validator.send_scoring_query(query, uid=assignment.uid)
            if response is not None:
                await self.scoring_store.save_response(assignment, response)
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
        validator = self.search_validators.get(search_type)
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
            assignments = await self._get_window_assignments(time_range_start)
            all_responses = await self.scoring_store.get_all_for_assignments(
                assignments,
                bucket_locators=(
                    await self._get_window_bucket_state(time_range_start)
                ).bucket_locators,
            )

            if not all_responses:
                bt.logging.warning(
                    "[QueryScheduler] No responses for epoch "
                    f"{time_range_start.isoformat()}, skipping scoring."
                )
                return

            score_tasks = [
                self._score_search_type(search_type, items, time_range_start)
                for search_type, items in all_responses.items()
                if self.search_validators.get(search_type) is not None and items
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
        neurons = getattr(self.neuron.metagraph, "neurons", None)
        if neurons is not None:
            return sorted(
                int(neuron.uid)
                for neuron in neurons
                if not getattr(neuron, "validator_permit", False)
            )

        metagraph_uids = self.neuron.metagraph.uids
        if hasattr(metagraph_uids, "tolist"):
            return [int(uid) for uid in metagraph_uids.tolist()]

        return [int(uid) for uid in metagraph_uids]

    def _store_window_assignments(
        self,
        time_range_start: datetime,
        assignments: list[ScoringAssignment],
    ) -> None:
        self.window_assignments[time_range_start] = tuple(assignments)

        cutoff = time_range_start - timedelta(hours=2)
        self.window_assignments = {
            current_window: items
            for current_window, items in self.window_assignments.items()
            if current_window >= cutoff
        }
        self.window_bucket_states = {
            current_window: items
            for current_window, items in self.window_bucket_states.items()
            if current_window >= cutoff
        }

    def _build_assignments_for_window(
        self,
        *,
        time_range_start: datetime,
        window_state: WindowSeedState,
    ) -> list[ScoringAssignment]:
        if window_state.validator_count == 0:
            return []

        assignments = build_scoring_assignments(
            time_range_start=time_range_start,
            miner_uids=self._collect_miner_uids(),
            validators=window_state.committed_validators,
            question_pool=self.question_pool,
            combined_seed=window_state.combined_seed,
        )
        self._store_window_assignments(time_range_start, assignments)
        return assignments

    async def _get_window_assignments(
        self,
        time_range_start: datetime,
    ) -> list[ScoringAssignment]:
        cached_assignments = self.window_assignments.get(time_range_start)
        if cached_assignments is not None:
            return list(cached_assignments)

        active_validators = await self.neuron.get_validators()
        window_state = await load_window_seed_state(
            subtensor=self.neuron.subtensor,
            netuid=self.neuron.config.netuid,
            validators=active_validators,
            time_range_start=time_range_start,
        )

        assignments = self._build_assignments_for_window(
            time_range_start=time_range_start,
            window_state=window_state,
        )

        if assignments:
            bt.logging.info(
                "[QueryScheduler] Rebuilt missing window assignments "
                f"time_range={time_range_start.isoformat()} "
                f"validators={window_state.validator_count} "
                f"assignments_total={len(assignments)}"
            )

        return assignments

    def _store_window_bucket_state(
        self,
        time_range_start: datetime,
        bucket_state: WindowBucketState,
    ) -> None:
        self.window_bucket_states[time_range_start] = bucket_state

    async def _get_window_bucket_state(
        self,
        time_range_start: datetime,
    ) -> WindowBucketState:
        cached_bucket_state = self.window_bucket_states.get(time_range_start)
        if cached_bucket_state is not None:
            return cached_bucket_state

        active_validators = await self.neuron.get_validators()
        bucket_state = await load_window_bucket_state(
            subtensor=self.neuron.subtensor,
            netuid=self.neuron.config.netuid,
            validators=active_validators,
            time_range_start=time_range_start,
        )
        self._store_window_bucket_state(time_range_start, bucket_state)
        return bucket_state

    async def _sleep_until_score_phase(self, time_range_start: datetime) -> None:
        score_phase_start = time_range_start + SCORING_STORAGE_PUBLISH_OFFSET
        delay = (score_phase_start - datetime.now(timezone.utc)).total_seconds()
        if delay > 0:
            await asyncio.sleep(delay)

    async def _run_window(self, time_range_start: datetime) -> None:
        active_validators = await self.neuron.get_validators()
        score_phase_start = time_range_start + SCORING_STORAGE_PUBLISH_OFFSET

        window_state = await build_window_seed_state(
            subtensor=self.neuron.subtensor,
            wallet=self.neuron.wallet,
            netuid=self.neuron.config.netuid,
            uid=self.neuron.uid,
            validators=active_validators,
            time_range_start=time_range_start,
        )

        if window_state.validator_count == 0:
            bt.logging.warning(
                "[QueryScheduler] No validator commitments available for "
                f"{time_range_start.isoformat()}, skipping window."
            )
            return

        assignments = self._build_assignments_for_window(
            time_range_start=time_range_start,
            window_state=window_state,
        )
        local_assignments = filter_scoring_assignments(
            assignments,
            validator_uid=self.neuron.uid,
        )

        bt.logging.info(
            "[QueryScheduler] Built local scoring plan "
            f"time_range={time_range_start.isoformat()} "
            f"validators={window_state.validator_count} "
            f"assignments_total={len(assignments)} "
            f"local_assignments={len(local_assignments)} "
            f"combined_seed={window_state.combined_seed}"
        )

        if local_assignments and datetime.now(timezone.utc) < score_phase_start:
            semaphore = asyncio.Semaphore(self.scoring_concurrency)

            tasks = [
                asyncio.create_task(self._run_assignment(assignment, semaphore))
                for assignment in local_assignments
            ]

            results = await asyncio.gather(*tasks, return_exceptions=True)

            for result in results:
                if isinstance(result, Exception):
                    bt.logging.error(f"[QueryScheduler] Window task failed: {result}")
        elif local_assignments:
            bt.logging.warning(
                "[QueryScheduler] Query phase already ended for "
                f"{time_range_start.isoformat()}, skipping late local scoring queries."
            )

        await self._sleep_until_score_phase(time_range_start)

        bucket_state = await build_window_bucket_state(
            subtensor=self.neuron.subtensor,
            wallet=self.neuron.wallet,
            netuid=self.neuron.config.netuid,
            uid=self.neuron.uid,
            validators=active_validators,
            time_range_start=time_range_start,
            bucket_locator=self.scoring_store.bucket_locator,
            publish_offset=SCORING_STORAGE_PUBLISH_OFFSET,
            reveal_delay_seconds=DEFAULT_BUCKET_COMMITMENT_REVEAL_DELAY_SECONDS,
        )
        self._store_window_bucket_state(time_range_start, bucket_state)

        if bucket_state.validator_count == 0:
            bt.logging.warning(
                "[QueryScheduler] No validator bucket commitments available for "
                f"{time_range_start.isoformat()}, skipping scoring."
            )
            return

        await self.score_epoch(time_range_start)

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
                self.current_time_range = window_start
                await self._run_window(window_start)
                await self._sleep_until_next_window(window_start)
            except Exception as e:
                bt.logging.error(
                    "[QueryScheduler] Unexpected scheduler error "
                    f"time_range={self.current_time_range.isoformat() if self.current_time_range else None}: {e}"
                )
                await asyncio.sleep(5)
