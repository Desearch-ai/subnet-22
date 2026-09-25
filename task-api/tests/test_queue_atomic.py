import asyncio
import json
import time

import redis.asyncio as aioredis
from app.queues import (
    QUEUE,
    VOPEN,
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


async def claimed(queue, hotkey="m", budget=5):
    got = await queue.claim(hotkey, budget)
    await queue.redis.set(f"issued:{got.task_id}", json.dumps({"key": f"k-{got.seq}"}))
    return got


async def refused(call):
    try:
        await call
    except Refusal as refusal:
        return refusal.code
    return None


def test_concurrent_claims_cannot_exceed_the_budget():
    async def scenario(redis):
        queue, _ = await filled(redis, 10)
        codes = await asyncio.gather(*(refused(queue.claim("m", 1)) for _ in range(10)))
        return codes.count(None), codes.count("NO_CAPACITY")

    assert run(scenario) == (1, 9)


def test_a_completed_task_can_never_return_to_the_queue():
    async def scenario(redis):
        queue, _ = await filled(redis)
        got = await claimed(queue)
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


def test_only_an_expired_claim_is_reclaimed():
    async def scenario(redis):
        queue, order = await filled(redis)
        got = await claimed(queue)
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
        got = await claimed(queue)
        return await queue.abandon(got.task_id, "rival"), await queue.abandon(
            got.task_id, "m"
        )

    rival, holder = run(scenario)
    assert rival is None and holder


def test_completion_needs_the_issued_key_and_a_live_claim():
    async def scenario(redis):
        queue, _ = await filled(redis)
        got = await claimed(queue)
        wrong_key = await queue.complete(got.task_id, "m", {}, "someone-else")
        await redis.zadd("claims:expiry", {got.task_id: 1})
        expired = await queue.complete(got.task_id, "m", {}, f"k-{got.seq}")
        return wrong_key, expired

    assert run(scenario) == (None, None)


def test_older_rounds_are_served_first():
    async def scenario(redis):
        queue, first = await filled(redis, 3, "r1")
        _, second = await filled(redis, 3, "r2")
        served = [(await queue.claim("m", 10)).task_id for _ in range(6)]
        return served, first + second

    served, expected = run(scenario)
    assert served == expected


def test_a_reclaimed_task_goes_back_ahead_of_newer_rounds():
    async def scenario(redis):
        queue, first = await filled(redis, 2, "r1")
        got = await claimed(queue)
        await filled(redis, 2, "r2")
        await queue.reclaim(got.task_id, now=10**12)
        return (await queue.claim("n", 10)).task_id, got.task_id

    served, reclaimed = run(scenario)
    assert served == reclaimed


def test_a_miner_is_never_given_back_a_task_it_held():
    async def scenario(redis):
        queue, order = await filled(redis, 2)
        got = await claimed(queue)
        await queue.reclaim(got.task_id, now=10**12)
        again = (await queue.claim("m", 5)).task_id
        other = (await queue.claim("n", 5)).task_id
        return got.task_id, again, other, order

    first, again, other, order = run(scenario)
    assert first == order[0] and again == order[1] and other == order[0]


def test_a_miner_that_held_every_waiting_task_is_told_so():
    async def scenario(redis):
        queue, _ = await filled(redis, 1)
        got = await claimed(queue)
        await queue.abandon(got.task_id, "m")
        try:
            await queue.claim("m", 5)
        except Refusal as refusal:
            return refusal.code, refusal.inputs

    assert run(scenario) == ("ALREADY_HELD", {"depth": 1, "held": 1})


async def validation_job(redis, task_id, miner="m", kind="crawl", at=None):
    job = {"task_id": task_id, "miner": miner, "kind": kind}
    await redis.set(f"vjob:{task_id}", json.dumps(job))
    await redis.zadd(VOPEN, {task_id: at or time.time()})
    await redis.sadd(f"inflight:{miner}", task_id)


def vote(validator: str, verdict: str = "pass") -> dict:
    return {"validator": validator, "verdict": verdict}


def test_one_vote_per_validator_per_open_upload():
    async def scenario(redis):
        validation = ValidationQueue(redis)
        await validation_job(redis, "a", at=1)
        await validation_job(redis, "b", at=2)
        listed = await validation.open_ids()
        first = await validation.vote("a", "v", vote("v"))
        twice = await validation.vote("a", "v", vote("v"))
        return (
            listed,
            first,
            twice,
            await validation.has_voted("a", "v"),
            await validation.has_voted("b", "v"),
            await validation.voters("a"),
        )

    listed, first, twice, seen_a, seen_b, voters = run(scenario)
    assert listed == ["a", "b"]
    assert first == 1 and twice == 0, "one vote per validator per upload"
    assert seen_a and not seen_b and voters == {"v"}


def test_reporting_marks_the_validator_active_for_the_window():
    async def scenario(redis):
        validation = ValidationQueue(redis, active_s=100)
        await validation_job(redis, "t")
        await validation.present("u", now=990)
        await validation.vote("t", "v", vote("v"), now=1000)
        await validation.vote("t", "w", vote("w"), now=1050)
        return (
            await validation.active(now=1080),
            await validation.active(now=1120),
            await validation.active(now=1200),
        )

    assert run(scenario) == ({"u", "v", "w"}, {"w"}, set())


def test_finalizing_closes_the_upload_and_hands_back_the_votes():
    async def scenario(redis):
        validation = ValidationQueue(redis)
        await validation_job(redis, "t")
        await validation.vote("t", "v", vote("v"))
        late = await validation.vote("t", "w", vote("w", "fail"))
        finalized = await validation.finalize("t")
        again = await validation.finalize("t")
        return (
            late,
            finalized,
            again,
            await validation.job("t"),
            await validation.depth(),
        )

    late, finalized, again, job, depth = run(scenario)
    assert late == 2
    assert finalized.job["task_id"] == "t"
    assert [v["validator"] for v in finalized.votes] == ["v", "w"]
    assert again is None and job is None and depth == 0


def test_the_oldest_open_upload_sets_the_backlog_age():
    async def scenario(redis):
        queue, _ = await filled(redis, 2)
        validation = ValidationQueue(redis)
        first = await claimed(queue)
        second = await claimed(queue)
        await queue.complete(
            first.task_id,
            "m",
            {"task_id": first.task_id, "miner": "m"},
            f"k-{first.seq}",
        )
        await redis.zadd(VOPEN, {first.task_id: 1})
        await queue.complete(
            second.task_id,
            "m",
            {"task_id": second.task_id, "miner": "m"},
            f"k-{second.seq}",
        )
        while_open = await validation.oldest_age()
        await validation.finalize(first.task_id)
        return while_open, await validation.oldest_age()

    while_open, after = run(scenario)
    assert while_open > 10**9 and after < 10


def test_a_pass_ends_validation_and_queues_publishing_in_one_step():
    async def scenario(redis):
        validation, publish = ValidationQueue(redis), PublishQueue(redis, 60)
        await validation_job(redis, "t")
        passed = await validation.finalize("t", {"task_id": "t", "key": "k"})
        in_flight = await redis.sismember("inflight:m", "t")
        return passed, await validation.job("t"), await publish.claim(5), in_flight

    passed, job, claimed, in_flight = run(scenario)
    assert passed.job["task_id"] == "t" and job is None
    assert claimed == [{"task_id": "t", "key": "k"}]
    assert not in_flight, "a decided task no longer counts against the miner's budget"


def test_an_unacked_publish_comes_back_and_an_acked_one_does_not():
    async def scenario(redis):
        validation, publish = ValidationQueue(redis), PublishQueue(redis, 60)
        for task_id in ("a", "b"):
            await validation_job(redis, task_id)
            await validation.finalize(task_id, {"task_id": task_id})
        first = await publish.claim(5)
        await publish.ack("a")
        await redis.zadd("publish:claims", {"b": 0})
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
            ValidationQueue(redis),
            PublishQueue(redis, 60, max_tries=2),
        )
        await validation_job(redis, "t")
        await validation.finalize("t", {"task_id": "t", "completed_at": 1000.0})
        waited = await publish.oldest_age()
        outcomes = []
        for _ in range(2):
            await publish.claim(1)
            await redis.zadd("publish:claims", {"t": 0})
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


def test_touching_a_publish_claim_extends_it():
    async def scenario(redis):
        validation, publish = ValidationQueue(redis), PublishQueue(redis, 600)
        await validation_job(redis, "t")
        await validation.finalize("t", {"task_id": "t"})
        await publish.claim(1)
        await redis.zadd("publish:claims", {"t": 5})
        await publish.extend_claim("t")
        return await redis.zscore("publish:claims", "t"), await publish.expired()

    score, expired = run(scenario)
    assert score > 10**9 and expired == []


def test_a_completed_task_counts_against_the_budget_until_its_verdict():
    async def scenario(redis):
        queue, _ = await filled(redis, 3)
        validation = ValidationQueue(redis)
        got = await claimed(queue, budget=1)
        await queue.complete(
            got.task_id, "m", {"task_id": got.task_id, "miner": "m"}, f"k-{got.seq}"
        )
        blocked = await refused(queue.claim("m", 1))
        await validation.finalize(got.task_id)
        freed = await refused(queue.claim("m", 1))
        return blocked, freed

    assert run(scenario) == ("NO_CAPACITY", None)


async def filled_kind(redis, kind, count=2, round_id="r1"):
    queue = TaskQueue(redis, 60, kind)
    order = [f"{kind}-{round_id}-t{n}" for n in range(count)]
    payloads = {t: {"urls": [f"https://x.example/{t}"]} for t in order}
    await queue.fill(round_id, order, payloads)
    return queue, order


def test_each_kind_serves_only_its_own_tasks_and_counts_its_own_budget():
    async def scenario(redis):
        crawl, crawl_tasks = await filled_kind(redis, "crawl")
        embed, embed_tasks = await filled_kind(redis, "embed")
        got_crawl = await crawl.claim("m", 1)
        got_embed = await embed.claim("m", 1)
        blocked = await refused(crawl.claim("m", 1))
        return got_crawl, got_embed, blocked, crawl_tasks, embed_tasks

    got_crawl, got_embed, blocked, crawl_tasks, embed_tasks = run(scenario)
    assert got_crawl.task_id == crawl_tasks[0] and got_crawl.payload["kind"] == "crawl"
    assert got_embed.task_id == embed_tasks[0] and got_embed.payload["kind"] == "embed"
    assert blocked == "NO_CAPACITY", "an embed task does not use up the crawl budget"


def test_an_expired_embed_task_goes_back_to_the_embed_queue():
    async def scenario(redis):
        crawl, _ = await filled_kind(redis, "crawl", 1)
        embed, order = await filled_kind(redis, "embed", 1)
        got = await embed.claim("m", 5)
        holder, _ = await embed.reclaim(got.task_id, now=10**12)
        return (
            holder,
            await redis.scard("inflight:embed:m"),
            await embed.depth(),
            await crawl.depth(),
        )

    assert run(scenario) == ("m", 0, 1, 1)


def test_a_miner_that_held_the_whole_front_of_the_queue_is_served_from_behind():
    async def scenario(redis):
        queue, order = await filled(redis, 60)
        for task_id in order[:55]:
            await redis.sadd(f"holders:{task_id}", "m")
        return (await queue.claim("m", 5)).task_id, order

    got, order = run(scenario)
    assert got == order[55]
