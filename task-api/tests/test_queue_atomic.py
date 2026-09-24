import asyncio
import json

import redis.asyncio as aioredis
from app.queues import (
    QUEUE,
    VALIDATE,
    VLEASES,
    PublishQueue,
    Refusal,
    TaskQueue,
    ValidationQueue,
)

REDIS_URL = "redis://localhost:6379/14"


def run(scenario):
    async def main():
        redis = aioredis.from_url(REDIS_URL, decode_responses=True)
        await redis.flushdb()
        try:
            return await scenario(redis)
        finally:
            await redis.flushdb()
            await redis.aclose()

    return asyncio.run(main())


async def filled(redis, count=3, round_id="r1"):
    queue = TaskQueue(redis, 60)
    order = [f"{round_id}-t{n}" for n in range(count)]
    payloads = {t: {"url_count": 1, "urls": [f"https://x.example/{t}"]} for t in order}
    await queue.fill(round_id, order, payloads)
    return queue, order


async def leased(queue, hotkey="m", budget=5):
    got = await queue.lease(hotkey, budget)
    await queue.redis.set(f"issued:{got.task_id}", json.dumps({"key": f"k-{got.seq}"}))
    return got


async def refused(call):
    try:
        await call
    except Refusal as refusal:
        return refusal.code
    return None


def test_concurrent_leases_cannot_exceed_the_budget():
    async def scenario(redis):
        queue, _ = await filled(redis, 10)
        codes = await asyncio.gather(*(refused(queue.lease("m", 1)) for _ in range(10)))
        return codes.count(None), codes.count("NO_CAPACITY")

    assert run(scenario) == (1, 9)


def test_a_completed_task_can_never_return_to_the_queue():
    async def scenario(redis):
        queue, _ = await filled(redis)
        got = await leased(queue)
        seq = await queue.complete(
            got.task_id, "m", {"task_id": got.task_id}, f"k-{got.seq}"
        )
        late = await queue.reclaim(got.task_id, now=10**12)
        again = await queue.abandon(got.task_id, "m")
        queued = await redis.zscore(QUEUE, got.task_id) is not None
        return seq, late, again, await queue.payload(got.task_id), queued

    seq, late, again, payload, queued = run(scenario)
    assert seq and late is None and again is None
    assert payload is None and not queued


def test_only_an_expired_lease_is_reclaimed():
    async def scenario(redis):
        queue, order = await filled(redis)
        got = await leased(queue)
        early = await queue.reclaim(got.task_id)
        late = await queue.reclaim(got.task_id, now=10**12)
        head = await redis.zrange(QUEUE, 0, 0)
        return early, late, head, order[0], await queue.in_flight("m")

    early, late, head, first, in_flight = run(scenario)
    assert early is None
    assert late[0] == "m"
    assert head == [first]
    assert in_flight == 0


def test_only_the_holder_can_abandon():
    async def scenario(redis):
        queue, _ = await filled(redis)
        got = await leased(queue)
        return await queue.abandon(got.task_id, "rival"), await queue.abandon(
            got.task_id, "m"
        )

    rival, holder = run(scenario)
    assert rival is None and holder


def test_completion_needs_the_issued_key_and_a_live_lease():
    async def scenario(redis):
        queue, _ = await filled(redis)
        got = await leased(queue)
        wrong_key = await queue.complete(got.task_id, "m", {}, "someone-else")
        await redis.zadd("leases:expiry", {got.task_id: 1})
        expired = await queue.complete(got.task_id, "m", {}, f"k-{got.seq}")
        return wrong_key, expired

    assert run(scenario) == (None, None)


def test_older_rounds_are_served_first():
    async def scenario(redis):
        queue, first = await filled(redis, 3, "r1")
        _, second = await filled(redis, 3, "r2")
        served = [(await queue.lease("m", 10)).task_id for _ in range(6)]
        return served, first + second

    served, expected = run(scenario)
    assert served == expected


def test_a_reclaimed_task_goes_back_ahead_of_newer_rounds():
    async def scenario(redis):
        queue, first = await filled(redis, 2, "r1")
        got = await leased(queue)
        await filled(redis, 2, "r2")
        await queue.reclaim(got.task_id, now=10**12)
        return (await queue.lease("n", 10)).task_id, got.task_id

    served, reclaimed = run(scenario)
    assert served == reclaimed


def test_a_miner_is_never_given_back_a_task_it_held():
    async def scenario(redis):
        queue, order = await filled(redis, 2)
        got = await leased(queue)
        await queue.reclaim(got.task_id, now=10**12)
        again = (await queue.lease("m", 5)).task_id
        other = (await queue.lease("n", 5)).task_id
        return got.task_id, again, other, order

    first, again, other, order = run(scenario)
    assert first == order[0] and again == order[1] and other == order[0]


def test_a_miner_that_held_every_waiting_task_is_told_so():
    async def scenario(redis):
        queue, _ = await filled(redis, 1)
        got = await leased(queue)
        await queue.abandon(got.task_id, "m")
        try:
            await queue.lease("m", 5)
        except Refusal as refusal:
            return refusal.code, refusal.inputs

    assert run(scenario) == ("ALREADY_HELD", {"depth": 1, "held": 1})


async def validation_job(redis, task_id, miner="m"):
    await redis.set(f"vjob:{task_id}", json.dumps({"task_id": task_id, "miner": miner}))
    await redis.rpush(VALIDATE, task_id)
    await redis.sadd(f"inflight:{miner}", task_id)


def test_a_job_whose_validators_keep_dying_is_settled_not_requeued():
    async def scenario(redis):
        validation = ValidationQueue(redis, 60, max_tries=2)
        await validation_job(redis, "t")
        tries = []
        for _ in range(2):
            await validation.lease("v")
            await redis.zadd(VLEASES, {"t": 0})
            tries.append(await validation.give_back("t"))
        return (
            tries,
            await validation.depth(),
            await validation.settle("t"),
            await validation.job("t"),
        )

    tries, depth, settled, left = run(scenario)
    assert tries == [1, 2]
    assert depth == 0
    assert settled.job == {"task_id": "t", "miner": "m"} and left is None


def test_a_released_job_goes_to_the_back():
    async def scenario(redis):
        validation = ValidationQueue(redis, 60)
        await validation_job(redis, "a")
        await validation_job(redis, "b")
        first, _ = await validation.lease("v")
        released = await validation.release("a", "v")
        second, _ = await validation.lease("v")
        return first["task_id"], released, second["task_id"]

    assert run(scenario) == ("a", ("requeued", None), "b")


def test_scoring_extends_the_lease_and_rejects_strangers():
    async def scenario(redis):
        validation = ValidationQueue(redis, 60)
        await validation_job(redis, "t")
        await validation.lease("v")
        stranger = await validation.begin("t", "other", 120)
        began = await validation.begin("t", "v", 120)
        await redis.zadd(VLEASES, {"t": 1})
        expired = await validation.begin("t", "v", 120)
        return stranger, began, expired

    stranger, began, expired = run(scenario)
    assert stranger is None
    assert began == {"task_id": "t", "miner": "m"}
    assert expired is None


def test_the_oldest_waiting_job_is_measured_across_hand_backs():
    async def scenario(redis):
        queue, _ = await filled(redis, 2)
        validation = ValidationQueue(redis, 60)
        first = await leased(queue)
        second = await leased(queue)
        await queue.complete(
            first.task_id,
            "m",
            {"task_id": first.task_id, "miner": "m"},
            f"k-{first.seq}",
        )
        await redis.zadd("vjobs:completed", {first.task_id: 1})
        await queue.complete(
            second.task_id,
            "m",
            {"task_id": second.task_id, "miner": "m"},
            f"k-{second.seq}",
        )
        claimed, _ = await validation.lease("v")
        await validation.release(claimed["task_id"], "v")
        while_waiting = await validation.oldest_age()
        claimed, _ = await validation.lease("v")
        await validation.finalize(claimed["task_id"], "v")
        return (
            first.task_id,
            claimed["task_id"],
            while_waiting,
            await validation.oldest_age(),
        )

    first, finished, while_waiting, after = run(scenario)
    assert while_waiting > 10**9
    assert finished != first
    assert after > 10**9


def test_a_restored_task_returns_at_its_rank():
    async def scenario(redis):
        queue, order = await filled(redis, 3)
        got = await leased(queue)
        await queue.complete(got.task_id, "m", {}, f"k-{got.seq}")
        seq = await queue.restore(
            got.task_id, {"urls": ["u"], "rank": got.payload["rank"]}
        )
        head = (await queue.lease("n", 5)).task_id
        return seq, head, got.task_id

    seq, head, restored = run(scenario)
    assert seq > 0 and head == restored


def test_a_pass_ends_validation_and_queues_publishing_in_one_step():
    async def scenario(redis):
        validation, publish = ValidationQueue(redis, 60), PublishQueue(redis, 60)
        await validation_job(redis, "t")
        await validation.lease("v")
        stranger = await validation.finalize("t", "other", {"task_id": "t"})
        passed = await validation.finalize("t", "v", {"task_id": "t", "key": "k"})
        in_flight = await redis.sismember("inflight:m", "t")
        return (
            stranger,
            passed,
            await validation.job("t"),
            await publish.claim(5),
            in_flight,
        )

    stranger, passed, job, claimed, in_flight = run(scenario)
    assert stranger is None and passed.job["task_id"] == "t" and job is None
    assert claimed == [{"task_id": "t", "key": "k"}]
    assert not in_flight, "a decided task no longer counts against the miner's budget"


def test_an_unacked_publish_comes_back_and_an_acked_one_does_not():
    async def scenario(redis):
        validation, publish = ValidationQueue(redis, 60), PublishQueue(redis, 60)
        for task_id in ("a", "b"):
            await validation_job(redis, task_id)
            await validation.lease("v")
            await validation.finalize(task_id, "v", {"task_id": task_id})
        first = await publish.claim(5)
        await publish.ack("a")
        await redis.zadd("publish:leases", {"b": 0})
        returned = [t for t in await publish.expired() if await publish.give_back(t)]
        pending = [await redis.zscore("publish:pending", t) for t in ("a", "b")]
        return len(first), returned, await publish.claim(5), pending

    first, returned, again, pending = run(scenario)
    assert (first, returned) == (2, ["b"])
    assert again == [{"task_id": "b"}]
    assert pending[0] is None and pending[1] is not None


def test_a_publish_that_keeps_failing_is_set_aside_and_stops_counting_as_backlog():
    async def scenario(redis):
        validation, publish = (
            ValidationQueue(redis, 60),
            PublishQueue(redis, 60, max_tries=2),
        )
        await validation_job(redis, "t")
        await validation.lease("v")
        await validation.finalize("t", "v", {"task_id": "t", "completed_at": 1000.0})
        waited = await publish.oldest_age()
        outcomes = []
        for _ in range(2):
            await publish.claim(1)
            await redis.zadd("publish:leases", {"t": 0})
            outcomes.append(await publish.give_back("t"))
        return (
            waited,
            outcomes,
            await publish.dead_count(),
            await publish.oldest_age(),
            await publish.claim(1),
        )

    waited, outcomes, dead, after, again = run(scenario)
    assert waited > 10**9 - 10**5
    assert outcomes == [1, -1]
    assert (dead, after, again) == (1, 0.0, [])


def test_touching_a_publish_lease_extends_it():
    async def scenario(redis):
        validation, publish = ValidationQueue(redis, 60), PublishQueue(redis, 600)
        await validation_job(redis, "t")
        await validation.lease("v")
        await validation.finalize("t", "v", {"task_id": "t"})
        await publish.claim(1)
        await redis.zadd("publish:leases", {"t": 5})
        await publish.extend_lease("t")
        return await redis.zscore("publish:leases", "t"), await publish.expired()

    score, expired = run(scenario)
    assert score > 10**9 and expired == []


def test_a_completed_task_counts_against_the_budget_until_its_verdict():
    async def scenario(redis):
        queue, _ = await filled(redis, 3)
        validation = ValidationQueue(redis, 60)
        got = await leased(queue, budget=1)
        await queue.complete(
            got.task_id, "m", {"task_id": got.task_id, "miner": "m"}, f"k-{got.seq}"
        )
        blocked = await refused(queue.lease("m", 1))
        await validation.lease("v")
        await validation.finalize(got.task_id, "v")
        freed = await refused(queue.lease("m", 1))
        return blocked, freed

    assert run(scenario) == ("NO_CAPACITY", None)


def test_a_validator_may_hold_only_so_many_jobs():
    async def scenario(redis):
        validation = ValidationQueue(redis, 60, max_leases=2)
        for task_id in "abc":
            await validation_job(redis, task_id)
        held = [(await validation.lease("v"))[0]["task_id"] for _ in range(2)]
        full = await refused(validation.lease("v"))
        other = (await validation.lease("w"))[0]["task_id"]
        await validation.finalize(held[0], "v")
        again = (await validation.lease("w")) is None and await validation.lease("v")
        return held, full, other, again

    held, full, other, again = run(scenario)
    assert held == ["a", "b"] and full == "LEASE_LIMIT" and other == "c"
    assert again is None


def test_a_job_handed_back_too_often_leaves_validation():
    async def scenario(redis):
        validation = ValidationQueue(redis, 60, max_releases=2)
        await validation_job(redis, "t")
        await redis.zadd("vjobs:completed", {"t": 1})
        await validation.lease("v")
        uncounted = await validation.release("t", "v", counted=False)
        await validation.lease("v")
        first = await validation.release("t", "v")
        await validation.lease("v")
        second = await validation.release("t", "v")
        backlog = await redis.zscore("vjobs:completed", "t")
        return uncounted, first, second, backlog, await validation.depth()

    uncounted, first, second, backlog, depth = run(scenario)
    assert uncounted == ("requeued", None) and first == ("requeued", None)
    assert second[0] == "exhausted" and second[1].job["task_id"] == "t"
    assert backlog is None and depth == 0


def test_an_audit_goes_to_another_validator_and_keeps_the_first_vote():
    async def scenario(redis):
        validation = ValidationQueue(redis, 60)
        await validation_job(redis, "t")
        await validation.lease("v")
        voted = await validation.vote(
            "t", "v", {"validator": "v", "verdict": "pass"}, 10**10
        )
        again = await validation.lease("v")
        audit, _ = await validation.lease("w")
        settled = await validation.finalize("t", "w")
        return (
            voted,
            again,
            audit["task_id"],
            settled.votes,
            await redis.zcard("vjobs:audits"),
        )

    voted, again, audit, votes, audits = run(scenario)
    assert voted and again is None and audit == "t"
    assert votes == [{"validator": "v", "verdict": "pass"}] and audits == 0


def test_a_lapsed_lease_frees_the_validators_slot():
    async def scenario(redis):
        validation = ValidationQueue(redis, 60, max_leases=1)
        await validation_job(redis, "a")
        await validation_job(redis, "b")
        await validation.lease("v")
        await redis.zadd(VLEASES, {"a": 0})
        await validation.give_back("a")
        return (await validation.lease("v"))[0]["task_id"]

    assert run(scenario) == "b"
