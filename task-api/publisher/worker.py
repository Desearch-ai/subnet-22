from __future__ import annotations

import asyncio
import contextlib
import io
import logging
import random
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime

import pyarrow as pa
import pyarrow.parquet as pq
from botocore.exceptions import ClientError

from desearch.extraction import looks_blocked
from publisher.records import (
    ROW_COLUMNS,
    build_record,
    record_key,
    to_zstd,
    record_version,
    publish_window,
)

log = logging.getLogger("publisher")

IDLE_DELAY_S = 2.0
RETRY_DELAY_S = 5.0
# Only bounds pathological contention: each lost race is progress.
PUT_ATTEMPTS = 12
RACED = {"PreconditionFailed", "412"}
MISSING = {"NoSuchKey", "404", "NotFound"}

# Full records ride along so a rebuild reads a few big files.
CHANGE_SCHEMA = pa.schema(
    [
        ("key", pa.string()),
        ("kind", pa.string()),
        ("previous_content_sha1", pa.string()),
        ("published_at", pa.string()),
        ("url", pa.string()),
        ("domain", pa.string()),
        ("doc_id", pa.string()),
        ("title", pa.string()),
        ("published", pa.string()),
        ("author", pa.string()),
        ("lang", pa.string()),
        ("text", pa.large_string()),
        ("fetched_at", pa.string()),
        ("content_sha1", pa.string()),
        ("source", pa.string()),
        ("captured_at", pa.string()),
        ("assigned_url", pa.string()),
        ("final_url", pa.string()),
        ("canonical", pa.string()),
        ("status", pa.int32()),
        ("page_type", pa.string()),
        ("description", pa.string()),
        ("json_ld_types", pa.list_(pa.string())),
        ("headings", pa.list_(pa.string())),
        ("text_sha256", pa.string()),
        ("task_id", pa.string()),
        ("miner", pa.string()),
    ]
)
RECORD_COLUMNS = [
    name
    for name in CHANGE_SCHEMA.names
    if name not in {"key", "kind", "previous_content_sha1", "published_at"}
]


class UploadGone(Exception):
    pass


class Publisher:
    def __init__(self, queue, temp, pages, workers: int = 32, batch: int = 20):
        self.queue = queue
        self.temp = temp
        self.pages = pages
        self.batch = batch
        self.pool = ThreadPoolExecutor(workers)

    async def run(self, stop: asyncio.Event, idle_exit: int = 0) -> None:
        idle = 0
        while not stop.is_set():
            try:
                done = await self.run_once()
            except Exception:
                log.exception("publish pass failed")
                await _sleep(stop, RETRY_DELAY_S)
                continue
            if done:
                idle = 0
                continue
            idle += 1
            if idle_exit and idle >= idle_exit:
                return
            await _sleep(stop, IDLE_DELAY_S)

    async def run_once(self) -> int | None:
        jobs = await self.queue.claim(self.batch)
        if not jobs:
            return None
        settled, changes = [], []
        for job in jobs:
            await self.queue.extend_lease(job["task_id"])
            try:
                done, failure = await asyncio.to_thread(self.publish, job)
            except UploadGone as gone:
                log.error("task=%s upload %s before publishing", job["task_id"], gone)
                await self.queue.mark_lost(job["task_id"])
                settled.append(job)
                continue
            except Exception:
                log.exception(
                    "task=%s publish failed; it will be retried", job["task_id"]
                )
                continue
            changes += done
            if failure:
                log.error(
                    "task=%s published %d pages, then %s; it will be retried",
                    job["task_id"],
                    len(done),
                    failure,
                )
                continue
            settled.append(job)
        # Written before any ack, including partly failed jobs.
        if changes:
            await asyncio.to_thread(
                self.write_changes, changes, await self.queue.next_seq()
            )
        for job in settled:
            await self.queue.ack(job["task_id"])
            with contextlib.suppress(Exception):
                await self.temp.delete(job["key"])
        log.info(
            "published %d tasks, %d pages new or changed", len(settled), len(changes)
        )
        return len(settled)

    def publish(self, job: dict) -> tuple[list[dict], str | None]:
        extra = {"IfMatch": job["etag"]} if job.get("etag") else {}
        try:
            body = self.temp.client.get_object(
                Bucket=self.temp.bucket, Key=self.temp.path(job["key"]), **extra
            )["Body"].read()
        except ClientError as exc:
            code = exc.response.get("Error", {}).get("Code")
            if code in MISSING:
                raise UploadGone("expired") from None
            if code in RACED:
                raise UploadGone("changed") from None
            raise

        window = publish_window(job)
        captured = datetime.now(UTC)
        assigned = set(job["urls"])
        chosen: dict[str, dict] = {}
        for row in pq.read_table(io.BytesIO(body), columns=ROW_COLUMNS).to_pylist():
            if not publishable(row, assigned):
                continue
            record = build_record(row, job["task_id"], job["miner"], window, captured)
            key = record_key(record)
            if key not in chosen or _rank(record) > _rank(chosen[key]):
                chosen[key] = record

        changes, failure = [], None
        for future in [self.pool.submit(self.put_latest, r) for r in chosen.values()]:
            try:
                change = future.result()
            except Exception as exc:
                failure = failure or f"{type(exc).__name__}: {exc}"
                continue
            if change:
                changes.append(change)
        return changes, failure

    def put_latest(self, record: dict) -> dict | None:
        key = record_key(record)
        version = record_version(record)
        for _ in range(PUT_ATTEMPTS):
            current = self.head(key)
            meta = current[1] if current else {}
            if current and meta.get("version") == version:
                # Same task and content: an earlier attempt wrote it before a crash.
                if meta.get("task-id") == record["task_id"]:
                    return _change(record, key, "replayed", meta)
                return None
            stored = (meta.get("fetched-at", ""), meta.get("version", ""))
            if current and stored > (record["fetched_at"], version):
                return None
            try:
                self.put(key, record, version, current[0] if current else None)
            except ClientError as exc:
                if exc.response.get("Error", {}).get("Code") in RACED:
                    time.sleep(random.uniform(0.01, 0.1))
                    continue
                raise
            return _change(record, key, "changed" if current else "new", meta)
        raise RuntimeError(f"{key} kept changing underneath the publisher")

    def head(self, key: str) -> tuple[str, dict] | None:
        try:
            found = self.pages.client.head_object(
                Bucket=self.pages.bucket, Key=self.pages.path(key)
            )
        except ClientError as exc:
            if exc.response.get("Error", {}).get("Code") in MISSING:
                return None
            raise
        return found["ETag"], found.get("Metadata", {})

    def put(self, key: str, record: dict, version: str, etag: str | None) -> None:
        condition = {"IfMatch": etag} if etag else {"IfNoneMatch": "*"}
        self.pages.client.put_object(
            Bucket=self.pages.bucket,
            Key=self.pages.path(key),
            Body=to_zstd(record),
            ContentType="application/zstd",
            Metadata={
                "version": version,
                "content-sha1": record["content_sha1"],
                "fetched-at": record["fetched_at"],
                "task-id": record["task_id"],
            },
            **condition,
        )

    def write_changes(self, changes: list[dict], seq: int) -> str:
        now = datetime.now(UTC)
        key = f"changes/dt={now:%Y-%m-%d}/{seq:012d}-{uuid.uuid4().hex[:8]}.parquet"
        sink = io.BytesIO()
        pq.write_table(
            pa.Table.from_pylist(changes, schema=CHANGE_SCHEMA),
            sink,
            compression="zstd",
        )
        self.pages.client.put_object(
            Bucket=self.pages.bucket,
            Key=self.pages.path(key),
            Body=sink.getvalue(),
            ContentType="application/vnd.apache.parquet",
        )
        return key

    def close(self) -> None:
        self.pool.shutdown(wait=True)


def publishable(row: dict, assigned: set[str]) -> bool:
    if row["error"] is not None or not row["text"] or row["url"] not in assigned:
        return False
    return not looks_blocked(row["status"], "", row["text"], row["title"] or "")


def _rank(record: dict) -> tuple:
    return (
        record["assigned_url"] == record["url"],
        record["fetched_at"],
        record_version(record),
    )


def _change(record: dict, key: str, kind: str, previous: dict) -> dict:
    return {
        "key": key,
        "kind": kind,
        "previous_content_sha1": previous.get("content-sha1", ""),
        "published_at": datetime.now(UTC).isoformat(timespec="seconds"),
        **{name: record[name] for name in RECORD_COLUMNS},
    }


async def _sleep(stop: asyncio.Event, seconds: float) -> None:
    with contextlib.suppress(asyncio.TimeoutError):
        await asyncio.wait_for(stop.wait(), seconds)
