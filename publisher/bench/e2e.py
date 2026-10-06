"""`publisher serve` end to end against S3-compatible storage and Redis, fed and checked with the task API's own code: e2e.py BINARY SAMPLE_DIR WORK_DIR [PYTHON_JSON]."""

from __future__ import annotations

import asyncio
import io
import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import boto3
import pyarrow.parquet as pq
import redis
import redis.asyncio as aioredis

from app.queues import PublishQueue, pack_job

ENDPOINT = os.environ.get("E2E_ENDPOINT", "http://127.0.0.1:19100")
REDIS_URL = os.environ.get("E2E_REDIS", "redis://127.0.0.1:6379/13")
KEY, SECRET = (
    os.environ.get("E2E_KEY", "e2e-user"),
    os.environ.get("E2E_SECRET", "e2e-secret-key"),
)
TEMP, PAGES, PREFIX = "e2e-temp", "e2e-pages", "tp/"


def s3():
    return boto3.client(
        "s3",
        endpoint_url=ENDPOINT,
        aws_access_key_id=KEY,
        aws_secret_access_key=SECRET,
        region_name="us-east-1",
    )


def environment(work: Path, **extra) -> dict:
    return {
        **os.environ,
        "CF_R2_ENDPOINT": ENDPOINT,
        "CF_R2_ACCESS_KEY_ID": KEY,
        "CF_R2_SECRET_ACCESS_KEY": SECRET,
        "CF_R2_REGION": "us-east-1",
        "CF_R2_BUCKET": TEMP,
        "TASK_API_R2_PREFIX": PREFIX,
        "CF_R2_PAGES_BUCKET": PAGES,
        "TASK_API_REDIS": REDIS_URL,
        "PUBLISHER_INDEX": str(work / "index"),
        "PUBLISHER_CACHE_MB": "256",
        **extra,
    }


def upload(client, sample: Path, jobs: list[dict]) -> None:
    for job in jobs:
        body = (sample / f"{job['source']}.parquet").read_bytes()
        found = client.put_object(Bucket=TEMP, Key=PREFIX + job["key"], Body=body)
        assert found["ETag"].strip('"') == job["etag"].strip('"'), (
            "storage computes the same entity tag R2 did"
        )


def enqueue(db: redis.Redis, jobs: list[dict]) -> None:
    """What the task API's finalize does for each upload that passed validation."""
    for job in jobs:
        publish = {k: v for k, v in job.items() if k != "source"}
        db.set(f"pjob:{job['task_id']}", pack_job(publish))
        db.rpush("publish:ready", job["task_id"])
        db.zadd("publish:pending", {job["task_id"]: job["completed_at"]})


def queue(call: str, *args):
    """One PublishQueue call on its own event loop, as the task API makes it."""

    async def run():
        client = aioredis.from_url(REDIS_URL, decode_responses=True)
        try:
            return await getattr(PublishQueue(client), call)(*args)
        finally:
            await client.aclose()

    return asyncio.run(run())


def json_object(client, bucket: str, key: str):
    return json.loads(client.get_object(Bucket=bucket, Key=key)["Body"].read())


def feed(
    client, bucket: str, prefix: str, name: str, first: int, last: int
) -> list[dict]:
    rows = []
    for seq in range(first, last + 1):
        index = json_object(client, bucket, f"{prefix}{name}/seq/{seq:012d}.json")
        body = client.get_object(Bucket=bucket, Key=prefix + index["key"])[
            "Body"
        ].read()
        table = pq.read_table(io.BytesIO(body))
        assert index["rows"] in (None, table.num_rows), (index, table.num_rows)
        rows += table.to_pylist()
    return rows


def serve(binary: str, work: Path, **extra) -> subprocess.Popen:
    log = open(work / "serve.log", "a")
    return subprocess.Popen(
        [binary, "serve"],
        env=environment(work, **extra),
        stdout=log,
        stderr=subprocess.STDOUT,
    )


def main() -> None:
    binary, sample, work = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3])
    work.mkdir(parents=True, exist_ok=True)
    client, db = s3(), redis.Redis.from_url(REDIS_URL, decode_responses=True)
    db.flushdb()
    for bucket in (TEMP, PAGES):
        try:
            client.create_bucket(Bucket=bucket)
        except client.exceptions.BucketAlreadyOwnedByYou:
            for page in client.get_paginator("list_objects_v2").paginate(Bucket=bucket):
                for item in page.get("Contents", []):
                    client.delete_object(Bucket=bucket, Key=item["Key"])
    jobs = [
        {**job, "source": job["task_id"]}
        for job in json.loads((sample / "jobs.json").read_text())
    ]
    upload(client, sample, jobs)
    enqueue(db, jobs)

    started = time.monotonic()
    assert serve(binary, work, PUBLISHER_IDLE_EXIT="2").wait(timeout=600) == 0
    first_s = time.monotonic() - started
    assert (queue("finished"), queue("waiting"), queue("depth")) == (40, 0, 0)
    assert db.zcard("publish:claims") == 0 and db.get("publish:lost") is None
    assert not client.list_objects_v2(Bucket=TEMP, Prefix=PREFIX + "submitted/").get(
        "KeyCount"
    ), "uploads deleted"
    latest = json_object(client, PAGES, "changes/latest.json")["seq"]
    changes = feed(client, PAGES, "", "changes", 1, latest)
    outcomes = feed(client, TEMP, PREFIX, "outcomes", 1, int(db.get("outcomes:seq")))
    reference = json.loads(Path(sys.argv[4]).read_text()) if len(sys.argv) > 4 else None
    summary = {
        "first_run_s": round(first_s, 1),
        "change_files": latest,
        "change_rows": len(changes),
        "kinds": sorted({c["kind"] for c in changes}),
        "outcomes": {
            k: sum(1 for o in outcomes if o["outcome"] == k)
            for k in ("published", "unchanged", "failed", "dropped")
        },
    }
    if reference:
        batch = reference["batches"][0]
        published = {c["key"]: c for c in changes}
        summary["keys_match_python"] = set(published) == {c["key"] for c in batch["changes"]} | {u["key"] for u in batch["unchanged"]}
        fields = ("url", "domain", "doc_id", "title", "text", "fetched_at", "content_sha1", "assigned_url", "headings", "task_id")
        summary["rows_match_python"] = sum(all(published[c["key"]][f] == c[f] for f in fields) for c in batch["changes"])
        summary["rows_compared"] = len(batch["changes"])
    snapshot = f"index/snapshots/{datetime.now(timezone.utc):%Y-%m-%d}.parquet"
    for _ in range(100):
        if client.list_objects_v2(Bucket=PAGES, Prefix=snapshot).get("KeyCount"):
            break
        time.sleep(0.1)
    found = client.get_object(Bucket=PAGES, Key=snapshot)["Body"].read()
    summary["snapshot_rows"] = pq.read_metadata(io.BytesIO(found)).num_rows

    taken = jobs[0]["task_id"]
    queue("withdraw", [taken], "e2e")
    assert serve(binary, work, PUBLISHER_IDLE_EXIT="1").wait(timeout=120) == 0
    latest_after = json_object(client, PAGES, "changes/latest.json")["seq"]
    removed = feed(client, PAGES, "", "changes", latest + 1, latest_after)
    dropped = [
        o
        for o in feed(
            client,
            TEMP,
            PREFIX,
            "outcomes",
            1,
            int(db.get("outcomes:seq")),
        )
        if o["outcome"] == "dropped"
    ]
    summary["withdrawn_removed_rows"] = len(removed)
    summary["withdrawn_kinds"] = sorted({c["kind"] for c in removed})
    summary["dropped_outcomes"] = len(dropped)

    again = [
        {**job, "task_id": job["task_id"] + "-again", "key": job["key"] + "-again"}
        for job in jobs[1:21]
    ]
    upload(client, sample, again)
    enqueue(db, again)
    acked_before = int(db.get("publish:acked"))
    process = serve(binary, work, PUBLISHER_BATCH="5")
    while int(db.get("publish:acked")) == acked_before:
        time.sleep(0.05)
    process.send_signal(signal.SIGTERM)
    assert process.wait(timeout=60) == 0
    summary["sigterm"] = {
        "acked": int(db.get("publish:acked")) - acked_before,
        "left_ready": db.llen("publish:ready"),
        "left_claimed": db.zcard("publish:claims"),
    }
    assert summary["sigterm"]["left_claimed"] == 0, "nothing claimed is left behind"
    assert summary["sigterm"]["acked"] + summary["sigterm"]["left_ready"] == 20
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
