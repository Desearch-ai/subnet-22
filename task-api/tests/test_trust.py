import asyncio

import pytest
from app import lifecycle, registry
from app.auth import Keypair
from app.queues import AUDITS, COMPLETED
from app.seeds import REVEAL_AFTER_BLOCKS, LocalSeeds

from tests.test_api_flow import Harness, _expect, _score, revealed


def run(memory, scenario):
    async def main():
        async with Harness(memory) as h:
            await h.enqueue()
            return await scenario(h)

    return asyncio.run(main())


async def judged(h, validator, verdict="pass", **score):
    job = (await validator.post("/v1/validation/lease"))["job"]
    body = {
        **_score(verdict, 3, "ok" if verdict == "pass" else "content_mismatch"),
        **score,
    }
    return job, await validator.post(f"/v1/validation/{job['task_id']}/score", body)


def test_a_verdict_drawn_for_audit_stands_once_a_second_validator_agrees(
    api_env, memory
):
    api_env.setenv("TASK_API_AUDIT_RATE", "1")

    async def scenario(h):
        task = await h.mine()
        _, first = await judged(h, h.validator)
        waiting = await h.status(task["task_id"])
        again = (await h.validator.post("/v1/validation/lease"))["job"]
        _, second = await judged(h, h.other_validator)
        report = await h.report((await h.view(task["task_id"]))["score"]["report_key"])
        return (
            first["verdict"],
            waiting,
            again,
            second,
            report,
            h.core.validations.audit_standing(),
        )

    first, waiting, again, second, report, standing = run(memory, scenario)
    assert (first, waiting, again) == ("audit", "awaiting_audit", None)
    assert (second["verdict"], second["credited"]) == ("pass", 3)
    assert [vote["verdict"] for vote in report["votes"]] == ["pass", "pass"]
    assert all(
        s == {"audits": 1, "disagreements": 0, "excluded": False}
        for s in standing.values()
    )


def test_a_disputed_verdict_goes_to_a_third_validator_and_the_minority_is_marked(
    api_env, memory
):
    api_env.setenv("TASK_API_AUDIT_RATE", "1")

    async def scenario(h):
        task = await h.mine()
        await judged(h, h.validator, "pass")
        _, tie = await judged(h, h.other_validator, "fail")
        _, final = await judged(h, h.third_validator, "fail")
        view = await h.view(task["task_id"])
        return (
            tie["verdict"],
            final,
            view["status"],
            h.core.validations.audit_standing(),
        )

    tie, final, status, standing = run(memory, scenario)
    assert tie == "audit"
    assert (final["verdict"], final["credited"]) == ("fail", 0)
    assert status == "queued", "the failed task goes back out"
    assert sorted(s["disagreements"] for s in standing.values()) == [0, 0, 1]


def test_a_validator_that_keeps_disagreeing_is_shut_out(api_env, memory):
    async def scenario(h):
        for _ in range(10):
            h.core.validations.record_audit(
                [h.other_validator.hotkey], [h.validator.hotkey]
            )
        await _expect(403, h.validator.post("/v1/validation/lease"))
        return (await h.other_validator.post("/v1/validation/lease"))["job"]

    assert run(memory, scenario) is None


def test_an_audit_nobody_picks_up_is_decided_by_the_votes_so_far(api_env, memory):
    api_env.setenv("TASK_API_AUDIT_RATE", "1")

    async def scenario(h):
        task = await h.mine()
        await judged(h, h.validator)
        await h.redis.zadd(AUDITS, {task["task_id"]: 0})
        concluded = await lifecycle.conclude_overdue_audits(h.core)
        status = await h.status(task["task_id"])
        return concluded, task["task_id"], status, await h.core.publish.depth()

    concluded, task_id, status, publishing = run(memory, scenario)
    assert concluded == [task_id] and status == "pass" and publishing == 1


def test_a_job_handed_back_too_often_is_voided_and_goes_back_out(api_env, memory):
    api_env.setenv("TASK_API_MAX_RELEASES", "2")

    async def scenario(h):
        task = await h.mine()
        release = f"/v1/validation/{task['task_id']}/release"
        await h.validator.post("/v1/validation/lease")
        first = await h.validator.post(release, {"reason": "provider"})
        await h.other_validator.post("/v1/validation/lease")
        second = await h.other_validator.post(release, {"reason": "provider"})
        backlog = await h.redis.zscore(COMPLETED, task["task_id"])
        miner = (await h.public.get(f"/v1/miners/{h.miner.hotkey}")).json()
        return first, second, backlog, await h.view(task["task_id"]), miner

    first, second, backlog, view, miner = run(memory, scenario)
    assert (first["status"], second["status"]) == ("queued_for_validation", "void")
    assert backlog is None, "a task nobody can judge no longer holds back leasing"
    assert (view["status"], view["score"]["reason"]) == ("queued", "unjudged")
    crawl = miner["pools"]["crawl"]
    assert (crawl["budget"], crawl["in_flight"]) == (1, 0)


def test_hand_backs_are_rate_limited_per_validator(api_env, memory):
    api_env.setenv("TASK_API_RELEASES_PER_HOUR", "1")

    async def scenario(h):
        task = await h.mine()
        await h.validator.post("/v1/validation/lease")
        await h.validator.post(
            f"/v1/validation/{task['task_id']}/release", {"reason": "provider"}
        )
        await _expect(429, h.validator.post("/v1/validation/lease"))
        job = (await h.other_validator.post("/v1/validation/lease"))["job"]
        return job["task_id"] == task["task_id"]

    assert run(memory, scenario)


def test_a_validator_may_hold_only_so_many_jobs(api_env, memory):
    api_env.setenv("TASK_API_VALIDATOR_LEASES", "1")

    async def scenario(h):
        await h.mine()
        await h.mine(h.rival)
        await h.validator.post("/v1/validation/lease")
        await _expect(429, h.validator.post("/v1/validation/lease"))
        return (await h.other_validator.post("/v1/validation/lease"))["job"] is not None

    assert run(memory, scenario)


def test_a_task_that_keeps_failing_is_dropped_after_its_last_attempt(api_env, memory):
    api_env.setenv("TASK_API_MAX_ATTEMPTS", "2")

    async def scenario(h):
        task = await h.mine()
        await judged(h, h.validator, "fail")
        retry = await h.mine(h.rival)
        await judged(h, h.validator, "fail")
        round_id = task["round_id"]
        outcomes = [line["outcome"] for line in h.core.log.entries(round_id)]
        still_open = await h.redis.scard(lifecycle.open_round_key(round_id))
        return (
            task,
            retry,
            await h.core.payload(task["task_id"]),
            outcomes,
            still_open,
        )

    task, retry, payload, outcomes, still_open = run(memory, scenario)
    assert retry["task_id"] == task["task_id"]
    assert payload is None and "dropped" in outcomes
    assert still_open == 1, "the other task of the round is still out"


def test_nothing_compared_is_void_and_confirmed_errors_are_paid_but_not_published(
    api_env, memory
):
    async def scenario(h):
        await h.mine()
        _, void = await judged(
            h, h.validator, **_score("pass", 3, outcome="unverifiable")
        )
        await h.mine(h.rival)
        _, errors = await judged(
            h, h.validator, **_score("pass", 3, outcome="errors_confirmed")
        )
        return void, errors, await h.core.publish.depth()

    void, errors, publishing = run(memory, scenario)
    assert (void["verdict"], void["credited"]) == ("void", 0)
    assert (errors["verdict"], errors["credited"]) == ("pass", 3)
    assert publishing == 0


def test_rounds_survive_a_restart_and_close_once_every_task_is_decided(api_env, memory):
    async def scenario():
        from app.main import create_app

        async with Harness(memory) as h:
            one = [{"host": "a.example", "url": "https://a.example/1"}]
            enqueued = await h.admin.post(
                "/v1/admin/enqueue", {"urls": one, "batch_target": 3}
            )
            restarted = create_app(h.redis).state.core
            pending = [r.round_id for r in restarted.rounds.unrevealed()]
            assert await revealed(restarted) == 1
            open_before = await lifecycle.close_finished(h.core)
            await h.mine()
            await judged(h, h.validator)
            closed = await lifecycle.close_finished(h.core)
            view = (await h.public.get(f"/v1/rounds/{enqueued['round_id']}")).json()
            return enqueued["round_id"], pending, open_before, closed, view

    round_id, pending, open_before, closed, view = asyncio.run(scenario())
    assert pending == [round_id]
    assert open_before == [] and closed == [round_id]
    assert view["closed_at"] is not None


def test_receipts_on_chain_need_a_real_key(monkeypatch):
    monkeypatch.setenv("TASK_API_REGISTRY", "chain")
    monkeypatch.delenv("TASK_API_KEY_URI", raising=False)
    with pytest.raises(RuntimeError, match="TASK_API_KEY_URI"):
        registry.receipt_key_from_env()

    monkeypatch.setenv("TASK_API_KEY_URI", "//receipts-test")
    expected = Keypair.create_from_uri("//receipts-test").ss58_address
    assert registry.receipt_key_from_env().ss58_address == expected


def test_the_registry_keeps_its_last_good_copy_when_the_chain_is_down(monkeypatch):
    chain = registry.ChainRegistry(22, "finney", ttl=0)
    answers = iter([{"hk": registry.Entry("hk", 1, True)}])

    def load():
        try:
            return next(answers)
        except StopIteration:
            raise ConnectionError("chain is down") from None

    monkeypatch.setattr(chain, "_load", load)

    async def scenario():
        first = await chain.lookup("hk")
        await asyncio.sleep(0.05)
        second = await chain.lookup("hk")
        await asyncio.sleep(0.05)
        return first, second, await chain.lookup("hk")

    first, second, third = asyncio.run(scenario())
    assert first == second == third == registry.Entry("hk", 1, True)


def test_local_seeds_wait_as_many_blocks_as_the_chain_does():
    seeds = LocalSeeds(block_seconds=0.01)

    async def scenario():
        target = await seeds.target_block()
        early = await seeds.seed_for(target)
        await asyncio.sleep(0.01 * (REVEAL_AFTER_BLOCKS + 1))
        seed = await seeds.seed_for(target)
        return early, seed, await seeds.seed_for(target)

    early, seed, again = asyncio.run(scenario())
    assert early is None
    assert seed is not None and seed == again
