import asyncio
import io
import json
import time

import pyarrow.parquet as pq
from app import lifecycle, outcomes, sampling
from app.canonical import canonicalize
from publisher.records import page_key
from publisher.worker import Publisher

from tests.test_api_flow import Harness, _score
from tests.test_trust import judged


def run(memory, scenario, task_urls: int = 3):
    async def main():
        async with Harness(memory) as h:
            await h.enqueue(task_urls=task_urls)
            return await scenario(h)

    return asyncio.run(main())


async def established(h, hotkey: str, passed_at: float | None = None) -> None:
    """A hotkey past its first checks, busy enough that the draw rarely picks it."""
    for n in range(sampling.NEW_HOTKEY_PASSES):
        await h.core.db(
            h.core.checks.record,
            hotkey,
            f"earlier-{n}",
            True,
            passed_at or time.time() - 3600,
        )
    await h.redis.set(f"uploads:{hotkey}:{sampling.hour_of()}", 10**9)
    h.core.check_share = 0.0


async def settled(h, task_id: str, timeout: float = 5.0) -> str:
    """Why the upload was picked for a check, or 'finalized' when it was not."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        await lifecycle.settle_seeded(h.core)
        job = await h.core.validation.job(task_id)
        if job is None:
            return "finalized"
        if job.get("picked"):
            return job["picked"]
        await asyncio.sleep(0.05)
    raise AssertionError("the upload's seed block never came")


def credits(h, hotkey: str) -> int:
    (total,) = h.core.sqlite.execute(
        "SELECT COALESCE(SUM(amount), 0) FROM credits WHERE hotkey = ?", (hotkey,)
    ).fetchone()
    return total


def test_new_hotkeys_and_rechecks_come_first_then_the_share_and_an_hourly_one():
    assert sampling.pick_reason(0.99, 1000, 9, 0) == sampling.NEW
    assert sampling.pick_reason(0.99, 1000, 10, 3) == sampling.RECHECK
    assert sampling.pick_reason(0.04, 1000, 10, 0, share=0.05) == sampling.DRAW
    assert sampling.pick_reason(0.06, 1000, 10, 0, share=0.05) is None
    assert sampling.pick_reason(0.3, 2, 10, 0, share=0.05) == sampling.DRAW, (
        "a hotkey with two uploads an hour is drawn half the time"
    )


def test_a_new_hotkey_has_its_upload_checked(api_env, memory):
    async def scenario(h):
        task = await h.mine()
        return await settled(h, task["task_id"])

    assert run(memory, scenario) == sampling.NEW


def test_an_upload_no_check_drew_is_paid_on_its_report_and_published(api_env, memory):
    async def scenario(h):
        await established(h, h.miner.hotkey)
        await h.core.db(
            h.core.checks.record, h.miner.hotkey, "with-errors", True, 0.0, 8, 0
        )
        task = await h.mine(errors=1)
        how = await settled(h, task["task_id"])
        view = await h.view(task["task_id"])
        return how, view, await h.core.publish.depth(), credits(h, h.miner.hotkey)

    how, view, publishing, paid = run(memory, scenario)
    assert how == "finalized"
    assert (view["status"], view["score"]["reason"]) == ("pass", "unchecked")
    assert view["score"]["credited"] == 3, "two pages, and one error at 9/10"
    assert publishing == 1 and paid == 3


def test_a_report_short_of_coverage_fails_without_a_check(api_env, memory):
    async def scenario(h):
        await established(h, h.miner.hotkey)
        task = (await h.miner.post("/v1/tasks/claim"))["tasks"][0]
        from tests.test_api_flow import _parquet, _upload

        body = _parquet(task, h.miner.hotkey)
        await _upload(h, task["upload"], body)
        await h.miner.post(
            f"/v1/tasks/{task['task_id']}/complete",
            {"key": task["upload"]["key"], "rows": 1, "ok": 1, "errors": 0},
        )
        await settled(h, task["task_id"])
        return await h.view(task["task_id"])

    view = run(memory, scenario)
    assert (view["status"], view["score"]["reason"]) == ("queued", "coverage")


def test_a_failed_check_takes_back_what_passed_since_the_last_pass(api_env, memory):
    async def scenario(h):
        hotkey = h.miner.hotkey
        await established(h, hotkey)
        unchecked = await h.mine()
        await settled(h, unchecked["task_id"])
        await h.core.db(h.core.checks.start_recheck, hotkey)
        checked = await h.mine()
        picked = await settled(h, checked["task_id"])
        _, verdict = await judged(h, h.validator, "fail", task_id=checked["task_id"])
        taken = await h.view(unchecked["task_id"])
        return (
            picked,
            verdict["verdict"],
            taken["score"]["verdict"],
            credits(h, hotkey),
            await h.core.publish.depth(),
            await h.core.db(h.core.checks.recheck_left, hotkey),
            await h.core.db(h.core.budgets.locked_until, hotkey),
        )

    picked, verdict, taken, paid, publishing, recheck, locked = run(
        memory, scenario, task_urls=2
    )
    assert (picked, verdict) == (sampling.RECHECK, "fail")
    assert taken == "withdrawn" and paid == -2, (
        "its credit is taken back, and the failed upload costs its URLs"
    )
    assert publishing == 2, "its publish job and the withdrawal"
    assert recheck == sampling.RECHECK_UPLOADS, "its next uploads are checked"
    assert locked is None, "one fail is not a penalty"


def test_two_fails_among_the_last_ten_checks_lock_out_and_wipe_the_day(api_env, memory):
    async def scenario(h):
        hotkey = h.miner.hotkey
        await established(h, hotkey)
        for _ in range(2):
            await h.core.db(h.core.checks.start_recheck, hotkey)
            task = await h.mine()
            await settled(h, task["task_id"])
            await judged(h, h.validator, "fail", task_id=task["task_id"])
        until = await h.core.db(h.core.budgets.locked_until, hotkey)
        return until, credits(h, hotkey)

    until, paid = run(memory, scenario, task_urls=2)
    assert until is not None and until - time.time() > 47 * 3600
    assert paid == 0


def test_a_checked_upload_that_overstates_its_report_fails(api_env, memory):
    async def scenario(h):
        task = await h.mine()
        _, verdict = await judged(
            h, h.validator, **_score("pass", 3, outcome="errors_confirmed")
        )
        return verdict, await h.view(task["task_id"])

    verdict, view = run(memory, scenario)
    assert verdict["verdict"] == "fail"
    assert view["score"]["reason"] == lifecycle.REPORTED_ROWS


def test_every_upload_of_a_locked_out_hotkey_is_checked(api_env, memory):
    async def scenario(h):
        await established(h, h.miner.hotkey)
        task = (await h.miner.post("/v1/tasks/claim"))["tasks"][0]
        from tests.test_api_flow import _parquet, _upload

        body = _parquet(task, h.miner.hotkey)
        await _upload(h, task["upload"], body)
        await h.core.db(h.core.budgets.lock_out, h.miner.hotkey, "crawl", 1, "test")
        await h.miner.post(
            f"/v1/tasks/{task['task_id']}/complete",
            {"key": task["upload"]["key"], "rows": 3, "ok": 3, "errors": 0},
        )
        return await settled(h, task["task_id"])

    assert run(memory, scenario) == lifecycle.LOCKED


def test_room_counts_the_queue_and_a_batch_sent_twice_is_queued_once(api_env, memory):
    async def scenario(h):
        h.core.queue_target = 10
        from tests.test_api_flow import URLS

        body = {"urls": URLS[:2], "batch_id": "b-1"}
        first = await h.admin.post("/v1/admin/enqueue", body)
        again = await h.admin.post("/v1/admin/enqueue", body)
        room = (await h.public.get("/v1/room")).json()
        return first, again, room, len(h.core.rounds.unrevealed())

    first, again, room, unrevealed = run(memory, scenario)
    assert first == again and unrevealed == 1
    assert room["room_tasks"] == 10 - room["queue"] - room["unrevealed"]
    assert room["unrevealed"] == 1 and not room["refusing"]


def test_a_task_dropped_after_its_last_attempt_reaches_the_outcome_feed(
    api_env, memory
):
    async def scenario(h):
        task = (await h.miner.post("/v1/tasks/claim"))["tasks"][0]
        job = {
            "kind": "crawl",
            "attempts": h.core.max_attempts - 1,
            "round_id": task["round_id"],
            "miner": h.miner.hotkey,
            "urls": task["urls"],
        }
        await lifecycle.requeue(h.core, task["task_id"], job, "fail")
        index = json.loads(
            h.core.storage.client.get_object(
                Bucket=h.core.storage.bucket,
                Key=h.core.storage.path(outcomes.seq_key(1)),
            )["Body"].read()
        )
        latest = json.loads(
            h.core.storage.client.get_object(
                Bucket=h.core.storage.bucket,
                Key=h.core.storage.path(outcomes.LATEST_KEY),
            )["Body"].read()
        )
        body = h.core.storage.client.get_object(
            Bucket=h.core.storage.bucket, Key=h.core.storage.path(index["key"])
        )["Body"].read()
        return task, latest, pq.read_table(io.BytesIO(body)).to_pylist()

    task, latest, rows = run(memory, scenario)
    assert latest == {"seq": 1}
    assert sorted(row["url"] for row in rows) == sorted(task["urls"])
    assert {row["outcome"] for row in rows} == {outcomes.DROPPED}


def test_a_withdrawal_takes_the_pages_down_and_tells_the_bot(api_env, memory):
    async def scenario(h):
        await established(h, h.miner.hotkey)
        task = await h.mine()
        await settled(h, task["task_id"])
        publisher = Publisher(h.core.publish, h.core.storage, h.core.pages, workers=2)
        try:
            await publisher.run_once()
            key = h.core.pages.path(page_key(canonicalize(task["urls"][0])))
            before = await h.core.pages.stat(page_key(canonicalize(task["urls"][0])))
            await lifecycle.take_back(h.core, h.miner.hotkey, 0, "test")
            await publisher.run_once()
            after = await h.core.pages.stat(page_key(canonicalize(task["urls"][0])))
        finally:
            publisher.close()
        last = int(await h.redis.get(outcomes.SEQ))
        index = json.loads(
            h.core.storage.client.get_object(
                Bucket=h.core.storage.bucket,
                Key=h.core.storage.path(outcomes.seq_key(last)),
            )["Body"].read()
        )
        body = h.core.storage.client.get_object(
            Bucket=h.core.storage.bucket, Key=h.core.storage.path(index["key"])
        )["Body"].read()
        return key, before, after, pq.read_table(io.BytesIO(body)).to_pylist(), task

    _, before, after, rows, task = run(memory, scenario)
    assert before is not None and after is None
    assert {row["outcome"] for row in rows} == {outcomes.DROPPED}
    assert sorted(row["url"] for row in rows) == sorted(task["urls"])


def test_an_unreadable_upload_is_not_published_and_its_urls_go_back(api_env, memory):
    async def scenario(h):
        await established(h, h.miner.hotkey)
        task = (await h.miner.post("/v1/tasks/claim"))["tasks"][0]
        from tests.test_api_flow import _upload

        await _upload(h, task["upload"], b"PAR1" + b"\0" * 64 + b"PAR1")
        await h.miner.post(
            f"/v1/tasks/{task['task_id']}/complete",
            {"key": task["upload"]["key"], "rows": 3, "ok": 3, "errors": 0},
        )
        await settled(h, task["task_id"])
        publisher = Publisher(h.core.publish, h.core.storage, h.core.pages, workers=2)
        try:
            done = await publisher.run_once()
        finally:
            publisher.close()
        index = json.loads(
            h.core.storage.client.get_object(
                Bucket=h.core.storage.bucket,
                Key=h.core.storage.path(outcomes.seq_key(1)),
            )["Body"].read()
        )
        body = h.core.storage.client.get_object(
            Bucket=h.core.storage.bucket, Key=h.core.storage.path(index["key"])
        )["Body"].read()
        return (
            done,
            await h.core.publish.lost_count(),
            task,
            pq.read_table(io.BytesIO(body)).to_pylist(),
        )

    done, lost, task, rows = run(memory, scenario)
    assert done == 1 and lost == 1
    assert {row["outcome"] for row in rows} == {outcomes.FAILED}
    assert sorted(row["url"] for row in rows) == sorted(task["urls"])

