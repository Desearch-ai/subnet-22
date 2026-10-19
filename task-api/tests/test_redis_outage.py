from __future__ import annotations

import asyncio
import time

from redis import exceptions as redis_errors

from app import lifecycle

from tests.test_api_flow import Harness, _expect
from tests.test_completion_timing import strikes


def redis_gone(*_, **__):
    raise redis_errors.ConnectionError("Connection refused")


def test_a_claim_while_redis_is_down_finds_no_tasks(api_env, memory):
    async def scenario():
        async with Harness(memory) as h:
            await h.enqueue()
            h.core.write_wait = redis_gone
            return await h.miner.post("/v1/tasks/claim")

    answer = asyncio.run(scenario())
    assert answer["tasks"] == []
    assert answer["refusal"]["code"] == "UNAVAILABLE"
    assert answer["refusal"]["inputs"]["retry_after"] > 0


def test_a_claim_on_a_read_only_redis_finds_no_tasks(api_env, memory):
    async def scenario():
        async with Harness(memory) as h:
            await h.enqueue()

            async def read_only(*_, **__):
                raise redis_errors.ReadOnlyError(
                    "You can't write against a read only replica."
                )

            h.core.tasks["crawl"].claim = read_only
            return await h.miner.post("/v1/tasks/claim")

    answer = asyncio.run(scenario())
    assert answer["tasks"] == [] and answer["refusal"]["code"] == "UNAVAILABLE"


def test_other_calls_are_asked_to_retry_while_redis_is_down(api_env, memory):
    async def scenario():
        async with Harness(memory) as h:
            h.core.write_wait = redis_gone
            await _expect(503, h.miner.post("/v1/tasks/0123456789abcdef/abandon"))

    asyncio.run(scenario())


def test_a_claim_held_across_a_restart_expires_without_a_strike(api_env, memory):
    async def scenario():
        async with Harness(memory) as h:
            await h.enqueue()
            await h.miner.post("/v1/tasks/claim")
            await asyncio.sleep(0.05)
            h.core.started_at = time.time()
            await asyncio.sleep(0.05)
            await h.rival.post("/v1/tasks/claim")
            later = time.time() + 3600
            reclaimed = await lifecycle.reclaim_expired(h.core, now=later)
            return reclaimed, strikes(h, h.miner.hotkey), strikes(h, h.rival.hotkey)

    reclaimed, miner_strikes, rival_strikes = asyncio.run(scenario())
    assert len(reclaimed) == 2, "both claims go back to the queue"
    assert miner_strikes == [], "the claim held through the restart is forgiven"
    assert rival_strikes == ["claim_expired"], "a claim taken after the restart is not"
