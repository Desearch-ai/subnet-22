import asyncio
import time

from app import flow, queues

from tests.test_api_flow import Harness


def test_the_publish_rate_is_measured_over_the_window_and_never_below_the_floor():
    rate = flow.PublishRate(lag_s=600, limit_s=1800)
    assert rate.per_second() == flow.RATE_FLOOR, "nothing measured yet"

    rate.note(1_000, 0.0)
    rate.note(1_600, 60.0)
    assert rate.per_second() == 10.0
    assert rate.room(in_system=5_000) == 1_000, (
        "ten minutes at 10 a second, less what is in"
    )
    assert not rate.overloaded(18_000) and rate.overloaded(18_001)

    rate.note(1_600, 60.0 + flow.WINDOW_S + 1)
    assert rate.per_second() == flow.RATE_FLOOR, (
        "a stalled publisher falls back to the floor"
    )


def test_the_bot_gets_no_room_and_miners_no_tasks_while_publishing_is_far_behind(
    api_env, memory
):
    api_env.setenv("TASK_API_PUBLISH_LAG_S", "60")
    api_env.setenv("TASK_API_PUBLISH_LAG_LIMIT_S", "120")

    async def scenario():
        async with Harness(memory) as h:
            await h.enqueue()
            open_room = (await h.public.get("/v1/room")).json()
            now = time.time()
            await h.redis.zadd(
                queues.PPENDING, {f"waiting-{n}": now for n in range(250)}
            )
            room = (await h.public.get("/v1/room")).json()
            claim = await h.miner.post("/v1/tasks/claim")
            return open_room, room, claim

    open_room, room, claim = asyncio.run(scenario())
    assert open_room["room_tasks"] > 0
    assert room["room_tasks"] == 0 and room["in_system"] >= 250
    assert claim["tasks"] == [] and claim["refusal"]["code"] == "VALIDATION_BACKLOG"
    assert claim["refusal"]["inputs"]["publish_waiting"] == 250
