from __future__ import annotations

import asyncio
import json
import time
from datetime import datetime, timedelta, timezone

from app import lifecycle, queues, rounds

from tests.test_api_flow import Harness, _expect, _parquet, _upload


def strikes(h: Harness, hotkey: str) -> list[str]:
    rows = h.core.budgets.db.execute(
        "SELECT reason FROM strikes WHERE hotkey = ?", (hotkey,)
    ).fetchall()
    return [reason for (reason,) in rows]


async def uploaded(h: Harness) -> tuple[dict, dict]:
    """A claimed task with its file uploaded, and the completion report for it."""
    await h.enqueue()
    task = (await h.miner.post("/v1/tasks/claim"))["tasks"][0]
    body = _parquet(task, h.miner.hotkey)
    await _upload(h, task["upload"], body)
    rows = len(task["urls"])
    report = {
        "key": task["upload"]["key"],
        "rows": rows,
        "ok": rows,
        "errors": 0,
        "bytes": len(body),
    }
    return task, report


def slow_copies(h: Harness, seconds: float) -> list[str]:
    """Every upload copy takes this long, as when storage is slow."""
    copied, copy = [], h.core.storage.copy

    async def slow(src, dst, etag=None, into=None):
        copied.append(src)
        await asyncio.sleep(seconds)
        return await copy(src, dst, etag, into)

    h.core.storage.copy = slow
    return copied


async def expire_at(h: Harness, task_id: str, at: float) -> None:
    await h.redis.zadd(queues.CLAIMS, {task_id: at}, xx=True)


def test_a_completion_that_arrived_in_time_counts_however_long_the_api_takes(
    api_env, memory
):
    async def scenario():
        async with Harness(memory) as h:
            task, report = await uploaded(h)
            task_id = task["task_id"]
            slow_copies(h, 1.0)
            sent = time.time()
            await expire_at(h, task_id, sent + 0.3)
            completing = asyncio.create_task(
                h.miner.post(f"/v1/tasks/{task_id}/complete", report)
            )
            await asyncio.sleep(0.6)
            reclaimed = await lifecycle.reclaim_expired(h.core)
            done = await completing

            job = await h.core.validation.job(task_id)
            assert reclaimed == [], "a claim being completed is not taken back"
            assert done["status"] == "open_for_validation"
            assert sent <= job["completed_at"] < sent + 0.3, "stamped on arrival"
            assert strikes(h, h.miner.hotkey) == []

    asyncio.run(scenario())


def test_a_completion_that_arrived_after_the_claim_ended_is_refused(api_env, memory):
    async def scenario():
        async with Harness(memory) as h:
            task, report = await uploaded(h)
            await expire_at(h, task["task_id"], time.time() - 1)
            await _expect(
                409, h.miner.post(f"/v1/tasks/{task['task_id']}/complete", report)
            )

    asyncio.run(scenario())


def test_a_file_written_after_the_completion_arrived_is_refused(api_env, memory):
    async def scenario():
        async with Harness(memory) as h:
            task, report = await uploaded(h)
            stored = h.core.storage.path(report["key"])
            later = datetime.now(timezone.utc) + timedelta(seconds=60)
            memory.memory.objects[(h.core.storage.bucket, stored)]["modified"] = later
            await _expect(
                409, h.miner.post(f"/v1/tasks/{task['task_id']}/complete", report)
            )
            assert await h.core.validation.job(task["task_id"]) is None

    asyncio.run(scenario())


def test_a_repeated_completion_gets_the_first_ones_answer(api_env, memory):
    async def scenario():
        async with Harness(memory) as h:
            task, report = await uploaded(h)
            path = f"/v1/tasks/{task['task_id']}/complete"
            copied = slow_copies(h, 0.5)
            first, again = await asyncio.gather(
                h.miner.post(path, report), h.miner.post(path, report)
            )
            later = await h.miner.post(path, report)

            assert first == again == later
            assert len(copied) == 1, "the upload was taken in once"
            await _expect(409, h.rival.post(path, report))

    asyncio.run(scenario())


def test_an_abandon_during_a_completion_waits_and_costs_nothing(api_env, memory):
    async def scenario():
        async with Harness(memory) as h:
            task, report = await uploaded(h)
            task_id = task["task_id"]
            slow_copies(h, 0.5)
            completing = asyncio.create_task(
                h.miner.post(f"/v1/tasks/{task_id}/complete", report)
            )
            await asyncio.sleep(0.1)
            await _expect(409, h.miner.post(f"/v1/tasks/{task_id}/abandon"))

            assert (await completing)["status"] == "open_for_validation"
            assert strikes(h, h.miner.hotkey) == []

    asyncio.run(scenario())


def test_a_claims_clock_starts_when_the_claim_is_handed_over(api_env, memory):
    async def scenario():
        async with Harness(memory) as h:
            await h.enqueue()
            record = h.core.record

            async def slow_record(*args, **kwargs):
                await asyncio.sleep(1.0)
                return await record(*args, **kwargs)

            h.core.record = slow_record
            task = (await h.miner.post("/v1/tasks/claim"))["tasks"][0]
            handed = time.time()
            expiry = await h.core.tasks["crawl"].claim_expiry(task["task_id"])

            assert expiry >= handed + h.core.claim_ttl - 0.2
            assert task["expires_at"] == expiry - rounds.UPLOAD_GRACE_S

    asyncio.run(scenario())


def test_an_upload_too_close_to_its_deadline_is_not_listed_for_validators(
    api_env, memory
):
    from tests.test_api_flow import open_list, opened

    async def scenario():
        async with Harness(memory) as h:
            task, report = await uploaded(h)
            await h.miner.post(f"/v1/tasks/{task['task_id']}/complete", report)
            listed = [(await opened(h))["task_id"]]
            job = await h.core.validation.job(task["task_id"])
            job["deadline"] = time.time() + lifecycle.CHECKABLE_LEFT_S - 30
            await h.redis.set(f"vjob:{task['task_id']}", json.dumps(job))
            later = [m["task_id"] for m in (await open_list(h))["uploads"]]
            return listed, later

    listed, later = asyncio.run(scenario())
    assert listed and later == []
