import asyncio
import json

import pytest
from app import lifecycle, registry
from app.auth import Keypair
from app.seeds import REVEAL_AFTER_BLOCKS, LocalSeeds

from desearch.client import TaskApiError
from tests.test_api_flow import (
    URLS,
    Harness,
    _expect,
    _score,
    opened,
    revealed,
    verifier,
)

VALIDATOR = Keypair.create_from_uri("//validator-api-test").ss58_address
OTHER = Keypair.create_from_uri("//validator-api-test-2").ss58_address


def run(memory, scenario, batch_target: int = 3):
    async def main():
        async with Harness(memory) as h:
            await h.enqueue(batch_target=batch_target)
            return await scenario(h)

    return asyncio.run(main())


async def judged(h, validator, verdict="pass", task_id="", **score):
    """Votes on the task, or on the oldest open upload, with the task's own URLs."""
    job = await opened(h, validator, want=task_id)
    returned = len(job["urls"])
    body = {
        **_score(verdict, returned, "ok" if verdict == "pass" else "content_mismatch"),
        **score,
    }
    for sample, url in zip(body["samples"], job["urls"], strict=False):
        sample["url"] = url
    return job, await validator.post(f"/v1/validation/{job['task_id']}/score", body)


def test_one_active_validator_finalizes_alone(api_env, memory):
    async def scenario(h):
        task = await h.mine()
        _, verdict = await judged(h, h.validator)
        report = await h.report((await h.view(task["task_id"]))["score"]["report_key"])
        return verdict, await h.status(task["task_id"]), report

    verdict, status, report = run(memory, scenario)
    assert (verdict["verdict"], verdict["credited"]) == ("pass", 3)
    assert status == "pass" and "votes" not in report


def test_a_second_active_validator_must_vote_before_a_task_finalizes(api_env, memory):
    async def scenario(h):
        await h.mine()
        await judged(h, h.validator)
        second = await h.mine(h.rival)
        _, pending = await judged(h, h.other_validator, task_id=second["task_id"])
        waiting = await h.status(second["task_id"])
        _, finalized = await judged(h, h.validator, task_id=second["task_id"])
        report = await h.report(
            (await h.view(second["task_id"]))["score"]["report_key"]
        )
        return pending, waiting, finalized, report, h.core.validations.audit_standing()

    pending, waiting, finalized, report, standing = run(memory, scenario)
    assert pending["verdict"] == "pending" and waiting == "voting"
    assert (finalized["verdict"], finalized["credited"]) == ("pass", 3)
    assert [v["verdict"] for v in report["votes"]] == ["pass", "pass"]
    assert all(
        s == {"audits": 1, "disagreements": 0, "excluded": False}
        for s in standing.values()
    )


def test_a_validator_that_asked_for_work_holds_final_verdict_until_it_votes(
    api_env, memory
):
    async def scenario(h):
        task = await h.mine()
        await opened(h, h.other_validator)
        _, pending = await judged(h, h.validator)
        waiting = await h.status(task["task_id"])
        _, finalized = await judged(h, h.other_validator, task_id=task["task_id"])
        return pending, waiting, finalized

    pending, waiting, finalized = run(memory, scenario)
    assert pending["verdict"] == "pending" and waiting == "voting"
    assert (finalized["verdict"], finalized["credited"]) == ("pass", 3)


async def three_active(h):
    """Three tasks, so that every validator has voted once and the third is open."""
    first = await h.mine()
    await judged(h, h.validator, task_id=first["task_id"])
    second = await h.mine(h.rival)
    await judged(h, h.other_validator, task_id=second["task_id"])
    await judged(h, h.validator, task_id=second["task_id"])
    third = await h.mine()
    await judged(h, h.third_validator, task_id=third["task_id"])
    return third


def test_a_disputed_task_is_decided_by_the_majority_and_the_minority_is_marked(
    api_env, memory
):
    async def scenario(h):
        task = await three_active(h)
        _, still = await judged(h, h.validator, "fail", task_id=task["task_id"])
        _, final = await judged(h, h.other_validator, task_id=task["task_id"])
        return (
            still["verdict"],
            final,
            await h.status(task["task_id"]),
            h.core.validations.audit_standing(),
        )

    still, final, status, standing = run(memory, scenario, batch_target=2)
    assert still == "pending", "three validators are active, so three votes"
    assert (final["verdict"], final["credited"], status) == ("pass", 2, "pass")
    assert standing[VALIDATOR]["disagreements"] == 1
    assert sorted(s["disagreements"] for s in standing.values()) == [0, 0, 1]


def test_two_validators_that_disagree_void_the_task(api_env, memory):
    async def scenario(h):
        await h.mine()
        await judged(h, h.validator)
        task = await h.mine(h.rival)
        await judged(h, h.other_validator, "fail", task_id=task["task_id"])
        _, void = await judged(h, h.validator, task_id=task["task_id"])
        view = await h.view(task["task_id"])
        return void["verdict"], view["status"], view["score"]["reason"]

    assert run(memory, scenario) == ("void", "queued", "validators_disagree")


def test_a_task_below_quorum_at_its_deadline_is_void_and_goes_back_out(api_env, memory):
    async def scenario(h):
        await h.mine()
        await judged(h, h.validator)
        task = await h.mine(h.rival)
        _, pending = await judged(h, h.other_validator, task_id=task["task_id"])
        early = await lifecycle.finalize_due(h.core)
        job = await h.core.validation.job(task["task_id"])
        await h.redis.set(f"vjob:{task['task_id']}", json.dumps({**job, "deadline": 0}))
        late = await lifecycle.finalize_due(h.core)
        view = await h.view(task["task_id"])
        miner = (await h.public.get(f"/v1/miners/{h.rival.hotkey}")).json()
        return pending["verdict"], early, late, view, miner

    pending, early, late, view, miner = run(memory, scenario)
    assert pending == "pending" and early == []
    assert late == [view["task_id"]]
    assert (view["status"], view["score"]["reason"]) == ("queued", "no_quorum")
    assert miner["coverage"] == {} and miner["transitions"] == []


def test_a_lowballed_pass_among_three_is_the_odd_one_out(api_env, memory):
    lowball = {
        "sampled": 2,
        "matched": 1,
        "mismatched": 1,
        "samples": [
            {"url": "", "outcome": "matched"},
            {"url": "", "outcome": "mismatched"},
        ],
    }

    async def scenario(h):
        task = await three_active(h)
        await judged(h, h.validator, task_id=task["task_id"])
        _, final = await judged(
            h, h.other_validator, task_id=task["task_id"], **lowball
        )
        return final, h.core.validations.audit_standing()

    final, standing = run(memory, scenario, batch_target=2)
    assert (final["verdict"], final["credited"]) == ("pass", 2)
    assert standing[OTHER]["disagreements"] == 1


def test_a_report_the_task_could_not_have_produced_is_refused(api_env, memory):
    async def scenario(h):
        task = await h.mine()
        await _expect(422, judged(h, h.validator, returned=0))
        return await h.status(task["task_id"])

    assert run(memory, scenario) == "open", "the refused report changed nothing"


def test_an_excluded_validator_can_neither_see_nor_score(api_env, memory):
    async def scenario(h):
        task = await h.mine()
        for _ in range(10):
            h.core.validations.record_audit([], [h.validator.hotkey])
        await _expect(403, h.validator.post("/v1/validation/open"))
        job = await opened(h, h.other_validator)
        score = f"/v1/validation/{task['task_id']}/score"
        await _expect(
            403, h.validator.post(score, _score("pass", 3, url=job["urls"][0]))
        )
        return await h.status(task["task_id"])

    assert run(memory, scenario) == "open"


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


def test_a_validators_own_timeout_costs_the_miner_nothing(api_env, memory):
    async def scenario(h):
        task = await h.mine()
        _, verdict = await judged(
            h,
            h.validator,
            "fail",
            reason="unscorable",
            returned=0,
            sampled=0,
            matched=0,
            mismatched=0,
            samples=[],
        )
        miner = (await h.public.get(f"/v1/miners/{h.miner.hotkey}")).json()
        return verdict, await h.view(task["task_id"]), miner

    verdict, view, miner = run(memory, scenario)
    assert (verdict["verdict"], verdict["credited"]) == ("void", 0)
    assert (view["status"], view["score"]["reason"]) == ("queued", "unscorable")
    assert miner["coverage"] == {} and miner["transitions"] == []


def test_two_reports_for_one_task_at_once_leave_one_verdict_and_its_report(
    api_env, memory
):
    async def scenario(h):
        task = await h.mine()
        job = await opened(h, h.validator)
        score = f"/v1/validation/{task['task_id']}/score"
        body = _score("pass", 3, url=job["urls"][0])
        raced = await asyncio.gather(
            h.validator.post(score, body),
            h.validator.post(score, body),
            return_exceptions=True,
        )
        view = await h.view(task["task_id"])
        report = await h.report(view["score"]["report_key"])
        health = (await h.public.get("/v1/health")).json()
        return raced, report, health

    raced, report, health = run(memory, scenario)
    outcomes = [r.status if isinstance(r, TaskApiError) else "ok" for r in raced]
    assert "ok" in outcomes and set(outcomes) - {"ok"} <= {409, 503}
    assert report["verdict"] == "pass"
    assert (health["verdicts"]["pass"], health["verdicts"]["fail"]) == (1, 0)


def test_a_final_verdict_recorded_before_redis_closed_the_upload_is_paid_once(
    api_env, memory
):
    async def scenario(h):
        task = await h.mine()
        job = await opened(h, h.validator)
        score = f"/v1/validation/{task['task_id']}/score"
        body = _score("pass", 3, url=job["urls"][0])
        real = h.core.validation.finalize

        async def lost(*_, **__):
            return None

        h.core.validation.finalize = lost
        pending = await h.validator.post(score, body)
        h.core.validation.finalize = real
        once = (await h.public.get(f"/v1/miners/{h.miner.hotkey}")).json()
        closed = await lifecycle.finalize_due(h.core)
        miner = (await h.public.get(f"/v1/miners/{h.miner.hotkey}")).json()
        return (
            task["task_id"],
            pending,
            once,
            closed,
            miner,
            await h.status(task["task_id"]),
        )

    task_id, pending, once, closed, miner, status = run(memory, scenario)
    assert pending["verdict"] == "pending"
    assert once["pools"]["crawl"]["verified"] == 3, "finalized before Redis lost it"
    assert closed == [task_id] and status == "pass"
    assert miner["pools"]["crawl"]["verified"] == 3, "closed later, not paid again"


def test_a_completion_receipt_names_the_block_the_upload_was_frozen_at(api_env, memory):
    async def scenario(h):
        task = await h.mine()
        job = await opened(h, h.validator)
        entries = h.core.log.entries(task["round_id"])
        completed = next(e for e in entries if e["outcome"] == "completed")
        ok, why = verifier._signatures(
            task["round_id"], entries, h.core.key.ss58_address
        )
        return completed, job, ok, why

    completed, job, ok, why = run(memory, scenario)
    assert job["seed_block"] == completed["block"] + REVEAL_AFTER_BLOCKS
    assert ok, why


def test_a_round_revealed_while_redis_was_down_is_filled_afterwards(api_env, memory):
    async def scenario():
        async with Harness(memory) as h:
            enqueued = await h.admin.post(
                "/v1/admin/enqueue", {"urls": URLS, "batch_target": 3}
            )
            real = h.core.redis.sadd

            async def down(*_, **__):
                raise ConnectionError("redis is down")

            h.core.redis.sadd = down
            with pytest.raises(ConnectionError):
                await revealed(h.core)
            h.core.redis.sadd = real
            unfilled = [r.round_id for r in h.core.rounds.unfilled()]
            closed = await lifecycle.close_finished(h.core)
            filled = await lifecycle.fill_missing(h.core)
            depth = await h.core.tasks["crawl"].depth()
            task = (await h.miner.post("/v1/tasks/claim"))["task"]
            return enqueued["round_id"], unfilled, closed, filled, depth, task

    round_id, unfilled, closed, filled, depth, task = asyncio.run(scenario())
    assert unfilled == [round_id] and closed == [], "not closed as served"
    assert filled == [round_id] and depth == 2
    assert task["round_id"] == round_id


def test_polling_after_a_round_closes_leaves_its_proof_intact(api_env, memory):
    async def scenario():
        async with Harness(memory) as h:
            one = [{"host": "a.example", "url": "https://a.example/1"}]
            enqueued = await h.admin.post(
                "/v1/admin/enqueue", {"urls": one, "batch_target": 3}
            )
            round_id = enqueued["round_id"]
            await revealed(h.core)
            await h.mine()
            await judged(h, h.validator)
            closed = await lifecycle.close_finished(h.core)
            anchored = h.core.log.anchored_root(round_id)
            idle = await h.miner.post("/v1/tasks/claim")
            log = (await h.public.get(f"/v1/rounds/{round_id}/log")).json()
            return round_id, closed, anchored, idle, log

    round_id, closed, anchored, idle, log = asyncio.run(scenario())
    assert closed == [round_id]
    assert idle["refusal"]["code"] == "QUEUE_EMPTY"
    assert idle["receipt"]["body"]["round_id"] == "", "refused outside any round"
    leaves = [verifier.canonical_json(entry) for entry in log["entries"]]
    assert verifier.merkle_root(leaves) == log["anchor_root"] == anchored


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
