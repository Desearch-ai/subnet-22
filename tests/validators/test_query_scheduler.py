from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from neurons.validators.scoring.query_scheduler import QueryScheduler


@pytest.mark.asyncio
async def test_score_epoch_extracts_prompts_from_responses_and_passes_epoch_start():
    scoring_store = SimpleNamespace(
        get_all_for_range=AsyncMock(
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
        utility_api=SimpleNamespace(),
        scoring_store=scoring_store,
        validators={"web_search": validator},
    )
    epoch_start = datetime(2026, 3, 14, 10, 0, tzinfo=timezone.utc)

    await scheduler.score_epoch(epoch_start)

    validator.compute_rewards_and_penalties.assert_awaited_once()
    kwargs = validator.compute_rewards_and_penalties.await_args.kwargs
    assert kwargs["scoring_epoch_start"] == epoch_start
    assert kwargs["prompts"] == ["what is bittensor", "what is tao"]
