from __future__ import annotations

import asyncio
import hashlib
import importlib.util
import io
import json
import re
import time
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import redis.asyncio as aioredis
import uvicorn
from app import lifecycle
from app.storage import PARQUET, Changed

from desearch.client import TaskApiClient, TaskApiError
from tests.http_client import HttpClient

ROOT = Path(__file__).resolve().parents[1]
REDIS_URL = "redis://localhost:6379/14"
MINER, RIVAL = "//miner-api-test", "//rival-api-test"
VALIDATOR, OTHER_VALIDATOR, THIRD_VALIDATOR = (
    "//validator-api-test",
    "//validator-api-test-2",
    "//validator-api-test-3",
)
ADMIN = "//admin-api-test"
URLS = [
    {"host": f"site{i}.example", "url": f"https://site{i}.example/page/{i}"}
    for i in range(6)
]

spec = importlib.util.spec_from_file_location(
    "verify_round", ROOT / "tools" / "verify_round.py"
)
verifier = importlib.util.module_from_spec(spec)
spec.loader.exec_module(verifier)


@asynccontextmanager
async def serving(app):
    server = uvicorn.Server(
        uvicorn.Config(
            app,
            host="127.0.0.1",
            port=0,
            lifespan="off",
            ws="none",
            forwarded_allow_ips="*",
            log_level="warning",
        )
    )
    running = asyncio.create_task(server.serve())
    while not server.started:
        if running.done():
            await running
        await asyncio.sleep(0.01)
    try:
        yield f"http://127.0.0.1:{server.servers[0].sockets[0].getsockname()[1]}"
    finally:
        server.should_exit = True
        await running


class Harness:
    def __init__(self, backend):
        self.backend = backend

    async def __aenter__(self) -> Harness:
        from app.main import create_app

        self.redis = aioredis.from_url(REDIS_URL, decode_responses=True)
        await self.redis.flushdb()
        self.app = create_app(self.redis)
        self.core = self.app.state.core
        # Two namespaces, so a result in the wrong bucket is caught.
        self.core.storage = self.backend.storage()
        self.core.pages = self.backend.storage(
            bucket=self.core.storage.bucket, prefix="pages-bucket/"
        )
        self.served = serving(self.app)
        self.url = await self.served.__aenter__()
        self.public = HttpClient(self.url)
        self.miner = TaskApiClient(self.url, MINER)
        self.rival = TaskApiClient(self.url, RIVAL)
        self.validator = TaskApiClient(self.url, VALIDATOR)
        self.other_validator = TaskApiClient(self.url, OTHER_VALIDATOR)
        self.third_validator = TaskApiClient(self.url, THIRD_VALIDATOR)
        self.admin = TaskApiClient(self.url, ADMIN)
        self.r2 = self.backend.http()
        return self

    async def __aexit__(self, *_) -> None:
        for client in (
            self.miner,
            self.rival,
            self.validator,
            self.other_validator,
            self.third_validator,
            self.admin,
        ):
            await client.aclose()
        await self.public.aclose()
        await self.r2.aclose()
        await self.served.__aexit__(None, None, None)
        await self.redis.flushdb()
        await self.redis.aclose()

    async def status(self, task_id: str) -> str:
        return (await self.public.get(f"/v1/tasks/{task_id}")).json()["status"]

    async def view(self, task_id: str) -> dict:
        return (await self.public.get(f"/v1/tasks/{task_id}")).json()

    async def report(self, key: str) -> dict:
        response = await self.r2.get(self.core.pages.presign_get(key, 60))
        assert response.status == 200, response.text
        return response.json()

    async def size(self, key: str) -> int | None:
        found = await self.core.storage.stat(key)
        return found[0] if found else None

    async def enqueue(self, urls=URLS, batch_target: int = 3) -> dict:
        enqueued = await self.admin.post(
            "/v1/admin/enqueue", {"urls": urls, "batch_target": batch_target}
        )
        await revealed(self.core)
        return enqueued

    async def mine(self, miner: TaskApiClient | None = None) -> dict:
        miner = miner or self.miner
        task = (await miner.post("/v1/tasks/lease"))["task"]
        body = _parquet(task, miner.hotkey)
        await _upload(self, task["upload"], body)
        await miner.post(
            f"/v1/tasks/{task['task_id']}/complete",
            {"key": task["upload"]["key"], "bytes": len(body)},
        )
        return task


async def revealed(core, timeout: float = 5.0) -> int:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not core.rounds.unrevealed():
            return await core.queue.depth()
        await lifecycle.reveal_pending(core)
        await asyncio.sleep(0.02)
    raise AssertionError("no round was revealed")


def test_crawl_task_round_trip(api_env, backend):
    asyncio.run(_round_trip(backend))


def test_rate_limited_refusal_carries_retry_after_and_seq(api_env, memory):
    api_env.setenv("TASK_API_POLL_RATE", "1")
    asyncio.run(_rate_limited(memory))


def test_a_locked_out_miner_is_refused_until_the_lockout_ends(api_env, memory):
    asyncio.run(_locked_out(memory))


async def _locked_out(backend) -> None:
    async with Harness(backend) as h:
        hotkey = h.miner.hotkey
        await h.core.db(h.core.budgets.strike, hotkey, "content_mismatch", "t1", 1)
        until = await h.core.db(h.core.budgets.strike, hotkey, "coverage", "t2", 2)

        answer = await h.miner.post("/v1/tasks/lease")
        refusal = answer["refusal"]
        assert refusal["code"] == "LOCKED_OUT" and answer["receipt"]
        assert refusal["inputs"]["until"] == round(until, 3)
        assert 11 * 3600 < refusal["inputs"]["retry_after"] <= 12 * 3600
        view = (await h.public.get(f"/v1/miners/{hotkey}")).json()
        assert view["locked_until"] == until
        rival = await h.rival.post("/v1/tasks/lease")
        assert rival["refusal"]["code"] == "QUEUE_EMPTY"


async def _round_trip(backend) -> None:
    async with Harness(backend) as h:
        await _scenario(h)


async def _scenario(h: Harness) -> None:
    enqueue = {"urls": URLS, "batch_target": 3}
    assert (await h.public.post("/v1/admin/enqueue", json=enqueue)).status == 401
    await _expect(403, h.miner.post("/v1/admin/enqueue", enqueue))
    enqueued = await h.admin.post("/v1/admin/enqueue", enqueue)
    round_id = enqueued["round_id"]
    assert enqueued["batches"] == 2
    listed = (await h.public.get("/v1/rounds")).json()["rounds"]
    assert [(r["round_id"], r["manifest_hash"], r["revealed"]) for r in listed] == [
        (round_id, enqueued["manifest_hash"], False)
    ]
    assert (await h.miner.post("/v1/tasks/lease"))["refusal"]["code"] == "QUEUE_EMPTY"
    assert await revealed(h.core) == 2

    await _expect(403, h.validator.post("/v1/tasks/lease"))
    task = (await h.miner.post("/v1/tasks/lease"))["task"]
    task_id, upload = task["task_id"], task["upload"]
    assert task["round_id"] == round_id
    assert len(task["urls"]) == 3
    assert re.fullmatch(
        rf"uploads/dt=[\d-]+/task={task_id}/{h.miner.hotkey}-\d+\.parquet",
        upload["key"],
    )
    assert upload["content_type"] == PARQUET
    assert upload["expires_at"] == task["expires_at"]
    assert await h.status(task_id) == "leased"

    complete = f"/v1/tasks/{task_id}/complete"
    report = {"key": upload["key"], "rows": 3, "ok": 3, "errors": 0, "bytes": 0}
    await _expect(409, h.rival.post(complete, report))
    forged = upload["key"].replace(h.miner.hotkey, h.rival.hotkey)
    await _expect(400, h.miner.post(complete, {**report, "key": forged}))
    await _expect(422, h.miner.post(complete, report))

    parquet = _parquet(task, h.miner.hotkey)
    mislabelled = await h.r2.put(
        upload["url"],
        data=parquet,
        headers={"Content-Type": "application/octet-stream"},
    )
    assert mislabelled.status == 403
    await _upload(h, upload, parquet)
    done = await h.miner.post(complete, {**report, "bytes": len(parquet)})
    assert done == {"task_id": task_id, "status": "queued_for_validation"}
    assert await h.status(task_id) == "queued_for_validation"
    assert await h.size(upload["key"]) is None

    unscored = await h.miner.post("/v1/tasks/lease")
    assert unscored["refusal"]["code"] == "NO_CAPACITY", (
        "unscored work counts against the budget"
    )

    swapped = _parquet({**task, "urls": ["https://evil.example/"] * 3}, h.miner.hotkey)
    await _upload(h, upload, swapped)

    score = f"/v1/validation/{task_id}/score"
    await _expect(403, h.miner.post("/v1/validation/lease"))
    await _expect(403, h.miner.post(score, _score("pass", 3)))

    job = (await h.validator.post("/v1/validation/lease"))["job"]
    assert job["task_id"] == task_id
    assert job["miner"] == h.miner.hotkey
    assert re.fullmatch(
        rf"submitted/dt=[\d-]+/task={task_id}/{h.miner.hotkey}-\d+-[0-9a-f]{{8}}\.parquet",
        job["key"],
    )
    assert job["urls"] == task["urls"]
    downloaded = await h.r2.get(job["download_url"])
    assert downloaded.body == parquet
    assert pq.read_table(io.BytesIO(downloaded.body)).num_rows == 3
    assert await h.status(task_id) == "validating"
    assert (await h.validator.post("/v1/validation/lease"))["job"] is None
    await _expect(409, h.other_validator.post(score, _score("pass", 3)))
    await _expect(422, h.validator.post(score, {**_score("pass", 3), "matched": 0}))

    too_long = [{"url": URLS[0]["url"], "miner_snippet": "x" * 501}]
    await _expect(422, h.validator.post(score, {**_score("pass", 3), "urls": too_long}))
    details = [
        {
            "url": URLS[0]["url"],
            "status": 200,
            "sampled": True,
            "miner_snippet": "Story",
        },
        {"url": URLS[1]["url"], "status": 404, "error": "http_4xx"},
    ]
    scored = await h.validator.post(
        score, {**_score("pass", 3), "credited": 999, "urls": details}
    )
    assert scored == {
        "task_id": task_id,
        "verdict": "pass",
        "credited": 3,
        "miner_budget": 2,
    }
    view = await h.view(task_id)
    summary = view["score"]
    assert view["status"] == "pass"
    assert [(u["url"], u["error"], u["miner_snippet"]) for u in view["urls"]] == [
        (URLS[0]["url"], None, "Story"),
        (URLS[1]["url"], "http_4xx", None),
    ]
    for query in ({"miner": h.miner.hotkey}, {"validator": h.validator.hotkey}):
        listed = (await h.public.get("/v1/tasks", params=query)).json()["tasks"]
        assert [t["task_id"] for t in listed] == [task_id]
    stored = h.core.validations.db.execute(
        "SELECT report FROM validations WHERE task_id = ?", (task_id,)
    ).fetchone()[0]
    assert "urls" not in json.loads(stored)
    assert "urls" not in await h.report(summary["report_key"])
    assert summary["returned"] == 3
    assert summary["validator"] == h.validator.hotkey
    assert await h.size(summary["report_key"]) is None
    assert await h.size(job["key"]) == len(parquet)
    assert await h.core.publish.depth() == 1

    from app.canonical import canonicalize
    from publisher.records import from_zstd, page_key
    from publisher.worker import Publisher

    publisher = Publisher(h.core.publish, h.core.storage, h.core.pages, workers=4)
    try:
        assert await publisher.run_once() == 1
    finally:
        publisher.close()
    assert await h.size(job["key"]) is None
    assert await h.core.publish.depth() == 0
    for url in task["urls"]:
        key = h.core.pages.path(page_key(canonicalize(url)))
        body = h.core.pages.client.get_object(Bucket=h.core.pages.bucket, Key=key)[
            "Body"
        ].read()
        assert from_zstd(body)["assigned_url"] == url
    assert (await h.report(summary["report_key"]))["verdict"] == "pass"

    second = (await h.miner.post("/v1/tasks/lease"))["task"]
    second_id = second["task_id"]
    second_parquet = _parquet(second, h.miner.hotkey)
    await _upload(h, second["upload"], second_parquet)
    second_report = {
        **report,
        "key": second["upload"]["key"],
        "bytes": len(second_parquet),
    }
    h.core.max_upload = 10
    await _expect(413, h.miner.post(f"/v1/tasks/{second_id}/complete", second_report))
    assert await h.size(second["upload"]["key"]) is None, (
        "an oversized upload is removed"
    )
    h.core.max_upload = 64_000_000
    await _upload(h, second["upload"], second_parquet)
    await h.miner.post(f"/v1/tasks/{second_id}/complete", second_report)

    second_score = f"/v1/validation/{second_id}/score"
    assert (await h.validator.post("/v1/validation/lease"))["job"][
        "task_id"
    ] == second_id
    await h.redis.zadd("vleases:expiry", {second_id: 0})
    assert await lifecycle.return_expired_validations(h.core) == [second_id]
    assert await h.status(second_id) == "queued_for_validation"
    await _expect(409, h.validator.post(second_score, _score("fail", 3)))
    job = (await h.other_validator.post("/v1/validation/lease"))["job"]
    assert job["task_id"] == second_id

    scored = await h.other_validator.post(
        second_score, _score("fail", 3, "content_mismatch")
    )
    assert scored == {
        "task_id": second_id,
        "verdict": "fail",
        "credited": 0,
        "miner_budget": 1,
    }
    view = await h.view(second_id)
    assert view["status"] == "queued", "a failed task goes back out for someone else"
    assert (view["score"]["verdict"], view["score"]["reason"]) == (
        "fail",
        "content_mismatch",
    )
    assert await h.size(view["score"]["upload_key"]) == len(second_parquet)
    assert (await h.report(view["score"]["report_key"]))["verdict"] == "fail"

    again = await h.mine(h.rival)
    assert (again["task_id"], again["urls"]) == (second_id, second["urls"])
    await h.validator.post("/v1/validation/lease")
    await h.validator.post(second_score, _score("pass", 3))

    drained = await h.miner.post("/v1/tasks/lease")
    assert drained["task"] is None
    assert drained["refusal"]["code"] == "QUEUE_EMPTY"

    health = (await h.public.get("/v1/health")).json()
    assert health["verdicts"] == {"pass": 2, "fail": 1}
    assert (
        health["queue_depth"],
        health["validation_depth"],
        health["validating"],
    ) == (0, 0, 0)
    miner = (await h.public.get(f"/v1/miners/{h.miner.hotkey}")).json()
    assert miner["verdicts"] == {"pass": 1, "fail": 1}
    assert (miner["budget"], miner["in_flight"]) == (1, 0)
    assert miner["coverage"]["returned"] == 6

    assert await lifecycle.close_finished(h.core) == [round_id]
    published = (await h.public.get(f"/v1/rounds/{round_id}")).json()
    entries = (await h.public.get(f"/v1/rounds/{round_id}/log")).json()
    ok, why = verifier._replay(
        published["manifest"], published["serve_order"], entries["entries"]
    )
    assert ok, why
    leaves = [verifier.canonical_json(line) for line in entries["entries"]]
    assert verifier.merkle_root(leaves) == entries["anchor_root"]
    signer = (await h.public.get("/v1/key")).json()["signer"]
    assert published["signer"] == signer == h.core.key.ss58_address
    ok, why = verifier._signatures(round_id, entries["entries"], signer)
    assert ok, why
    assert [line["outcome"] for line in entries["entries"]] == [
        "issued",
        "completed",
        "refused",
        "issued",
        "completed",
        "reclaimed",
        "issued",
        "completed",
        "refused",
    ]
    reclaimed = [line for line in entries["entries"] if line["outcome"] == "reclaimed"]
    assert [line["cause"] for line in reclaimed] == ["fail"]
    assert all(line["seq"] > 0 for line in entries["entries"])


async def _rate_limited(backend) -> None:
    async with Harness(backend) as h:
        refusals, receipts = [], []
        while len(refusals) < 5:
            answer = await h.miner.post("/v1/tasks/lease")
            refusals.append(answer["refusal"])
            receipts.append(answer["receipt"])
            if refusals[-1]["code"] == "RATE_LIMITED":
                break

        limited = refusals[-1]
        assert limited["code"] == "RATE_LIMITED"
        assert 0 < limited["inputs"]["retry_after"] <= 1
        assert receipts.pop() is None, "an excess poll is not signed into the log"
        assert all(r["inputs"]["retry_after"] == 10.0 for r in refusals[:-1])

        seqs = [line["seq"] for line in h.core.log.entries("")]
        assert len(seqs) == len(receipts)
        assert all(seqs) and len(set(seqs)) == len(seqs)
        ok, why = verifier._receipts(
            "", receipts, h.core.log.entries(""), h.core.key.ss58_address
        )
        assert ok, why
        forged = {**receipts[0], "body": {**receipts[0]["body"], "outcome": "issued"}}
        ok, _ = verifier._receipts(
            "", [forged], h.core.log.entries(""), h.core.key.ss58_address
        )
        assert not ok


async def _expect(status: int, call) -> None:
    with pytest.raises(TaskApiError) as caught:
        await call
    assert caught.value.status == status, caught.value.detail


async def _upload(h: Harness, upload: dict, body: bytes) -> None:
    response = await h.r2.put(
        upload["url"], data=body, headers={"Content-Type": upload["content_type"]}
    )
    assert response.status == 200, response.text


def _score(
    verdict: str, returned: int, reason: str = "ok", outcome: str | None = None
) -> dict:
    outcome = outcome or ("matched" if verdict == "pass" else "mismatched")
    return {
        "returned": returned,
        "missing": 0,
        "duplicates": 0,
        "sampled": 1,
        "matched": int(outcome == "matched"),
        "mismatched": int(outcome == "mismatched"),
        "unverifiable": int(outcome == "unverifiable"),
        "errors_confirmed": int(outcome == "errors_confirmed"),
        "errors_unconfirmed": 0,
        "reextract_mismatch": 0,
        "error_rows": returned if outcome == "errors_confirmed" else 0,
        "verdict": verdict,
        "reason": reason,
        "samples": [
            {
                "url": URLS[0]["url"],
                "outcome": outcome,
                "similarity": 0.93 if outcome == "matched" else 0.12,
                "miner_status": 200,
                "validator_status": 200,
                "miner_chars": 40,
                "validator_chars": 41,
            }
        ],
    }


def _parquet(task: dict, hotkey: str) -> bytes:
    from desearch.extraction.schema import PAGE_SCHEMA

    fetched_at = datetime.now(UTC)
    rows = []
    for url in task["urls"]:
        text = f"Hello from {url}"
        html = (
            f"<html><head><title>{url}</title></head><body><p>{text}</p></body></html>"
        )
        rows.append(
            {
                "url": url,
                "final_url": url,
                "status": 200,
                "error": None,
                "fetched_at": fetched_at,
                "elapsed_ms": 12,
                "content_type": "text/html",
                "html_bytes": len(html),
                "html": html.encode(),
                "html_sha256": hashlib.sha256(html.encode()).hexdigest(),
                "page_type": "other",
                "title": url,
                "description": "",
                "lang": "en",
                "canonical": url,
                "published": "",
                "author": "",
                "json_ld_types": [],
                "headings": [],
                "text": text,
                "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
            }
        )

    metadata = {
        **(PAGE_SCHEMA.metadata or {}),
        b"task_id": task["task_id"].encode(),
        b"hotkey": hotkey.encode(),
    }
    table = pa.Table.from_pylist(rows, schema=PAGE_SCHEMA.with_metadata(metadata))
    sink = io.BytesIO()
    pq.write_table(table, sink, compression="zstd")
    return sink.getvalue()


def test_a_copy_is_refused_when_the_source_changed_since_its_etag(backend):
    async def scenario():
        storage = backend.storage()
        await asyncio.to_thread(
            storage.client.put_object,
            Bucket=storage.bucket,
            Key=storage.path("src"),
            Body=b"validated bytes",
        )
        _, etag = await storage.stat("src")
        await asyncio.to_thread(
            storage.client.put_object,
            Bucket=storage.bucket,
            Key=storage.path("src"),
            Body=b"swapped bytes",
        )
        with pytest.raises(Changed):
            await storage.copy("src", "dst", etag)
        assert await storage.stat("dst") is None
        _, fresh = await storage.stat("src")
        await storage.copy("src", "dst", fresh)
        assert (await storage.stat("dst"))[0] == len(b"swapped bytes")

    asyncio.run(scenario())


def test_the_token_can_publish_into_the_real_pages_bucket(backend):
    if not backend.real:
        pytest.skip("checks the real pages bucket")
    from desearch import env
    from tests.memory_r2 import DOTENV

    pages_bucket = env.read_dotenv(DOTENV).get("CF_R2_PAGES_BUCKET") or "desearch-pages"

    async def scenario():
        pages = backend.storage(bucket=pages_bucket)
        try:
            await asyncio.to_thread(pages.client.head_bucket, Bucket=pages_bucket)
        except Exception as exc:
            pytest.skip(f"{pages_bucket} is not reachable with this token: {exc}")
        await asyncio.to_thread(
            pages.client.put_object,
            Bucket=pages_bucket,
            Key=pages.path("probe"),
            Body=b"v1",
            IfNoneMatch="*",
        )
        assert (await pages.stat("probe"))[0] == 2

    asyncio.run(scenario())


def test_races_backlog_storage_faults_and_voids(api_env, backend):
    asyncio.run(_faults(backend))


async def _faults(backend) -> None:
    async with Harness(backend) as h:
        await _fault_scenario(h)


async def _fault_scenario(h: Harness) -> None:
    await h.enqueue()
    task = (await h.miner.post("/v1/tasks/lease"))["task"]
    task_id = task["task_id"]
    parquet = _parquet(task, h.miner.hotkey)
    await _upload(h, task["upload"], parquet)
    complete = f"/v1/tasks/{task_id}/complete"
    report = {"key": task["upload"]["key"], "rows": 3, "ok": 3, "errors": 0, "bytes": 0}

    raced = await asyncio.gather(
        h.miner.post(complete, report),
        h.miner.post(complete, report),
        return_exceptions=True,
    )
    assert sorted(type(r).__name__ for r in raced) == ["TaskApiError", "dict"]
    job = await h.core.validation.job(task_id)
    assert await h.size(job["key"]) == len(parquet)

    h.core.max_backlog = 0.001
    await asyncio.sleep(0.2)
    held = await h.miner.post("/v1/tasks/lease")
    assert held["refusal"]["code"] == "VALIDATION_BACKLOG"
    h.core.max_backlog = 43_200

    score = f"/v1/validation/{task_id}/score"
    assert (await h.validator.post("/v1/validation/lease"))["job"]["task_id"] == task_id

    async def broken(*_):
        raise RuntimeError("pages bucket is down")

    h.core.pages.put_json = broken
    await _expect(502, h.validator.post(score, _score("pass", 3)))
    del h.core.pages.put_json
    assert await h.core.validation.lease_holder(task_id) is None
    assert await h.redis.get(f"vtries:{task_id}") is None
    assert await h.redis.get(f"vreleases:{task_id}") is None, (
        "our outage is not the task's"
    )
    assert await h.core.validation.depth() == 1

    assert (await h.validator.post("/v1/validation/lease"))["job"]["task_id"] == task_id
    await h.core.storage.delete(job["key"])
    released = await h.validator.post(
        f"/v1/validation/{task_id}/release", {"reason": "missing"}
    )
    assert released["status"] == "void"
    view = await h.view(task_id)
    assert (view["status"], view["score"]["reason"]) == ("queued", "upload_missing")
    assert (await h.report(view["score"]["report_key"]))["verdict"] == "void"
    assert (await h.public.get("/v1/health")).json()["verdicts"]["void"] == 1

    skipped = (await h.miner.post("/v1/tasks/lease"))["task"]
    assert skipped["task_id"] != task_id, "a miner never gets a task twice"
    again = (await h.rival.post("/v1/tasks/lease"))["task"]
    assert again["task_id"] == task_id
    assert again["urls"] == task["urls"]
    assert (await h.public.get(f"/v1/miners/{h.miner.hotkey}")).json()["budget"] == 1


def test_public_reads_are_limited_per_ip_and_listed_a_page_at_a_time(api_env, memory):
    api_env.setenv("TASK_API_READS_PER_MINUTE", "4")
    asyncio.run(_reads(memory))


async def _reads(backend) -> None:
    from app.validations import build_report

    async with Harness(backend) as h:
        for n in range(7):
            job = {
                "round_id": "r",
                "miner": "m1" if n % 2 else "m2",
                "key": "k",
                "urls": [],
            }
            report = build_report(
                f"t{n}", job, "v", {"verdict": "pass", "reason": "ok"}
            )
            h.core.validations.record({**report, "scored_at": float(n)})

        def client(ip: str) -> HttpClient:
            return HttpClient(h.url, headers={"X-Forwarded-For": ip})

        async with client("1.1.1.1") as one, client("2.2.2.2") as two:
            first = (await one.get("/v1/tasks", params={"limit": 3})).json()
            assert [t["task_id"] for t in first["tasks"]] == ["t6", "t5", "t4"]
            page = {"limit": 3, "before": first["next"]}
            second = (await one.get("/v1/tasks", params=page)).json()
            assert [t["task_id"] for t in second["tasks"]] == ["t3", "t2", "t1"]
            mine = (await one.get("/v1/tasks", params={"miner": "m1"})).json()
            assert [t["task_id"] for t in mine["tasks"]] == ["t5", "t3", "t1"]
            assert mine["next"] is None
            assert (await one.get("/v1/tasks", params={"limit": 101})).status == 422

            refused = await one.get("/v1/shares")
            assert refused.status == 429
            assert 0 < int(refused.headers["Retry-After"]) <= 60
            assert (await two.get("/v1/shares")).status == 200
