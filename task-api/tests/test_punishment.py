import asyncio
import time

from app import lifecycle
from app.budget import CRAWL, HOSTILE_LOCKOUT_H, HOUR, LOCKOUT_STEPS_H
from app.queues import CLAIMS

from tests.test_api_flow import Harness, _expect, _parquet, _upload
from tests.test_trust import judged


def run(memory, scenario, task_urls: int = 1):
    async def main():
        async with Harness(memory) as h:
            await h.enqueue(task_urls=task_urls)
            return await scenario(h)

    return asyncio.run(main())


async def lapse_all(h) -> None:
    await h.redis.zadd(
        CLAIMS, {task: 0 for task in await h.redis.zrange(CLAIMS, 0, -1)}
    )
    await lifecycle.reclaim_expired(h.core)


async def with_budget(h, budget: int) -> None:
    await h.core.db(h.core.budgets.get_or_create, h.miner.hotkey)
    h.core.sqlite.execute("UPDATE miners SET budget = ?", (budget,))
    h.core.sqlite.commit()


async def credit_of(h, hotkey: str) -> int:
    (total,) = h.core.sqlite.execute(
        "SELECT COALESCE(SUM(amount), 0) FROM credits WHERE hotkey = ?", (hotkey,)
    ).fetchone()
    return total


def test_a_hotkey_that_only_hoards_is_locked_out_on_its_second_lapse(api_env, memory):
    async def scenario(h):
        await h.miner.post("/v1/tasks/claim", {"count": 1})
        await lapse_all(h)
        first = await h.core.db(h.core.budgets.locked_until, h.miner.hotkey)
        await h.redis.flushdb()
        h.core.sqlite.execute("UPDATE strikes SET at = at - 600")
        h.core.sqlite.commit()
        await h.enqueue(task_urls=1)
        await h.miner.post("/v1/tasks/claim")
        await lapse_all(h)
        second = await h.core.db(h.core.budgets.locked_until, h.miner.hotkey)
        return first, second, await credit_of(h, h.miner.hotkey)

    first, second, credit = run(memory, scenario)

    assert first is None, "one lapse is a warning"
    assert second is not None and second - LOCKOUT_STEPS_H[0] * HOUR > 0
    assert credit == -2, "each lapse takes its URLs back"


def test_claims_that_lapse_together_cost_one_strike(api_env, memory):
    async def scenario(h):
        await with_budget(h, 5)
        claimed = await h.miner.post("/v1/tasks/claim", {"count": 5})
        await lapse_all(h)
        strikes = h.core.sqlite.execute("SELECT COUNT(*) FROM strikes").fetchone()[0]
        return len(claimed["tasks"]), strikes

    tasks, strikes = run(memory, scenario)
    assert tasks > 1 and strikes == 1


def test_an_abandoned_task_is_a_lapse_too(api_env, memory):
    async def scenario(h):
        task = (await h.miner.post("/v1/tasks/claim"))["tasks"][0]
        answer = await h.miner.post(f"/v1/tasks/{task['task_id']}/abandon")
        reasons = [
            row[0] for row in h.core.sqlite.execute("SELECT reason FROM strikes")
        ]
        return answer, reasons, await credit_of(h, h.miner.hotkey)

    answer, reasons, credit = run(memory, scenario)
    assert answer["budget"] == 1 and reasons == ["abandoned"] and credit == -1


def test_a_failed_task_takes_its_urls_back_from_the_day(api_env, memory):
    async def scenario(h):
        await h.mine()
        await judged(h, h.validator, "fail")
        return await credit_of(h, h.miner.hotkey)

    assert run(memory, scenario, task_urls=3) == -3


def test_an_upload_that_crashes_most_checks_locks_the_miner_out(api_env, memory):
    async def scenario(h):
        await h.mine()
        await judged(h, h.validator, "fail", reason="unscorable", crashed=True)
        until = await h.core.db(h.core.budgets.locked_until, h.miner.hotkey, CRAWL)
        claim = await h.miner.post("/v1/tasks/claim")
        return until - time.time(), claim["refusal"]["code"]

    left, code = run(memory, scenario, task_urls=3)
    assert HOSTILE_LOCKOUT_H * HOUR - 60 < left <= HOSTILE_LOCKOUT_H * HOUR
    assert code == "LOCKED_OUT"


def test_a_crash_the_validator_could_not_reproduce_costs_nothing(api_env, memory):
    async def scenario(h):
        await h.mine()
        await judged(h, h.validator, "fail", reason="unscorable")
        return await h.core.db(h.core.budgets.locked_until, h.miner.hotkey, CRAWL)

    assert run(memory, scenario, task_urls=3) is None


def test_an_upload_that_is_not_parquet_is_refused_at_completion(api_env, memory):
    async def scenario(h):
        task = (await h.miner.post("/v1/tasks/claim"))["tasks"][0]
        upload = task["upload"]
        await _upload(h, upload, b"<html>not a file of rows</html>")
        complete = f"/v1/tasks/{task['task_id']}/complete"
        await _expect(422, h.miner.post(complete, {"key": upload["key"]}))
        await _upload(h, upload, _parquet(task, h.miner.hotkey))
        return await h.miner.post(complete, {"key": upload["key"]})

    done = run(memory, scenario)
    assert done["status"] == "open_for_validation", "a fixed upload is still accepted"


def test_one_claim_returns_several_tasks_and_a_receipt_for_each(api_env, memory):
    async def scenario(h):
        await with_budget(h, 3)
        answer = await h.miner.post("/v1/tasks/claim", {"count": 5})
        return answer

    answer = run(memory, scenario)
    assert len(answer["tasks"]) == 3, "as many as asked, up to the budget"
    assert [r["body"]["task_id"] for r in answer["receipts"]] == [
        t["task_id"] for t in answer["tasks"]
    ]


def test_a_repeated_refusal_is_answered_but_signed_only_once_a_minute(api_env, memory):
    async def scenario(h):
        await h.redis.flushdb()
        first = await h.miner.post("/v1/tasks/claim")
        again = await h.miner.post("/v1/tasks/claim")
        return first, again

    first, again = run(memory, scenario)
    assert first["refusal"]["code"] == again["refusal"]["code"] == "QUEUE_EMPTY"
    assert first["receipt"] and again["receipt"] is None
