from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from neurons.validators.service.query_scheduler import QueryScheduler
from neurons.validators.service.scoring_dataset import (
    ScoringAssignment,
    ScoringQuestion,
)
from neurons.validators.service.seed_commitment import WindowBucketState


@pytest.mark.asyncio
async def test_score_epoch_extracts_prompts_from_responses_and_passes_epoch_start():
    scoring_store = SimpleNamespace(
        get_all_for_assignments=AsyncMock(
            return_value={
                "web_search": [
                    {
                        "uid": 11,
                        "response": {"query": "what is bittensor", "result": "a"},
                    },
                    {
                        "uid": 12,
                        "response": {"query": "what is tao", "result": "b"},
                    },
                ]
            }
        )
    )
    validator = SimpleNamespace(compute_rewards_and_penalties=AsyncMock())
    scheduler = QueryScheduler(
        neuron=SimpleNamespace(),
        scoring_store=scoring_store,
        validators={"web_search": validator},
    )
    epoch_start = datetime(2026, 3, 14, 10, 0, tzinfo=timezone.utc)
    scheduler.window_assignments[epoch_start] = (
        ScoringAssignment(
            time_range_start=epoch_start,
            uid=11,
            search_type="web_search",
            validator_uid=1,
            validator_hotkey="validator-a",
            question=ScoringQuestion(query="what is bittensor"),
            scoring_seed=101,
        ),
        ScoringAssignment(
            time_range_start=epoch_start,
            uid=12,
            search_type="web_search",
            validator_uid=1,
            validator_hotkey="validator-a",
            question=ScoringQuestion(query="what is tao"),
            scoring_seed=202,
        ),
    )
    scheduler.window_bucket_states[epoch_start] = WindowBucketState(
        committed_buckets=(
            SimpleNamespace(
                uid=1,
                hotkey="validator-a",
                bucket_locator="local:validator-a",
            ),
        )
    )

    await scheduler.score_epoch(epoch_start)

    scoring_store.get_all_for_assignments.assert_awaited_once()
    assert (
        scoring_store.get_all_for_assignments.await_args.kwargs["bucket_locators"]
        == {(1, "validator-a"): "local:validator-a"}
    )
    validator.compute_rewards_and_penalties.assert_awaited_once()
    kwargs = validator.compute_rewards_and_penalties.await_args.kwargs
    assert kwargs["scoring_epoch_start"] == epoch_start
    assert kwargs["prompts"] == ["what is bittensor", "what is tao"]
