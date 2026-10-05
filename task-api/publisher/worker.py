from __future__ import annotations

import asyncio
import contextlib
import hashlib
import io
import logging
import os
import tempfile
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

import pyarrow as pa
import pyarrow.parquet as pq
from botocore.exceptions import ClientError

from desearch.embedding import (
    INPUT_SCHEMA,
    OUTPUT_SCHEMA,
    read_parquet,
    write_parquet,
)
from app import outcomes
from app.feeds import Feed
from app.canonical import domain_of
from desearch.extraction import looks_blocked
from engine.chunking import doc_full, doc_head, para_chunks
from publisher.index import VersionIndex
from publisher.reading import RangeFile, read_rows
from publisher.records import (
    ROW_COLUMNS,
    build_record,
    record_key,
    record_version,
    publish_window,
)

log = logging.getLogger("publisher")

IDLE_DELAY_S = 2.0
RETRY_DELAY_S = 5.0
RACED = {"PreconditionFailed", "412"}
MISSING = {"NoSuchKey", "404", "NotFound"}
PARQUET = "application/vnd.apache.parquet"
EMBED_INPUT_PAGES = 500
# Small enough that one page's record is a single ranged read away.
CHANGE_ROW_GROUP = 1000
CHANGES = Feed("changes")
# One file per embed task: every text with its page and model, so the engine needs nothing else.
VECTORS_SCHEMA = pa.schema(
    [*INPUT_SCHEMA, ("model", pa.string()), *OUTPUT_SCHEMA.remove(0)]
)

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
        ("validator", pa.string()),
        ("validators", pa.list_(pa.string())),
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
    def __init__(
        self,
        queue,
        temp,
        pages,
        workers: int = 32,
        batch: int = 20,
        embed_inputs: bool = False,
        index: VersionIndex | None = None,
    ):
        self.queue = queue
        self.temp = temp
        self.pages = pages
        self.batch = batch
        self.embed_inputs = embed_inputs
        self.index = index or VersionIndex(":memory:")
        self.pool = ThreadPoolExecutor(workers)
        self.snapshot_day = ""

    async def run(self, stop: asyncio.Event, idle_exit: int = 0) -> None:
        idle = 0
        while not stop.is_set():
            try:
                done = await self.run_once()
                await self.snapshot_daily()
                await CHANGES.fill_holes(self.pages, self.queue.redis)
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
        finalized, chosen, failed = [], {}, []
        withdrawn = [
            task_id
            for job in jobs
            if job.get("kind") == "withdraw"
            for task_id in job["task_ids"]
        ]
        if withdrawn:
            await asyncio.to_thread(self.index.withdraw, withdrawn)
        removals = await asyncio.to_thread(self.index.of_tasks, withdrawn)
        reading = []
        for job in jobs:
            await self.queue.extend_claim(job["task_id"])
            if job.get("kind") == "withdraw":
                finalized.append(job)
            elif await asyncio.to_thread(self.index.is_withdrawn, job["task_id"]):
                failed += _failed(job, job["urls"])
                finalized.append(job)
            else:
                reading.append(job)
        # Each upload is read on its own thread, so a batch costs about one upload's reads.
        read = await asyncio.gather(
            *(self.on_pool(self.read_job, job) for job in reading),
            return_exceptions=True,
        )
        for job, outcome in zip(reading, read):
            if isinstance(outcome, UploadGone):
                log.error(
                    "task=%s upload %s before publishing", job["task_id"], outcome
                )
                await self.queue.mark_lost(job["task_id"])
                if job.get("kind") != "embed":
                    failed += _failed(job, job["urls"])
                finalized.append(job)
                continue
            if isinstance(outcome, BaseException):
                log.error(
                    "task=%s publish failed; it will be retried: %r",
                    job["task_id"],
                    outcome,
                )
                continue
            records, missed = outcome
            for record in records:
                key = record_key(record)
                if key not in chosen or _rank(record) > _rank(chosen[key]):
                    chosen[key] = record
            failed += missed
            finalized.append(job)

        changes, unchanged = await asyncio.to_thread(self.decide, chosen)
        # A withdrawn page that this batch replaces with a newer version is replaced, not removed.
        replaced = {change["key"] for change in changes}
        removals = [removal for removal in removals if removal[0] not in replaced]
        removed = [_removal(key, url, sha1) for key, url, _, sha1 in removals]
        # The change file is written before the index learns of it, so a crash only replays.
        if changes or removed:
            key = await asyncio.to_thread(self.write_changes, changes + removed)
            seq = await CHANGES.number(
                self.pages, self.queue.redis, key, len(changes) + len(removed)
            )
            await asyncio.to_thread(
                self.index.store,
                [_indexed(change, seq, row) for row, change in enumerate(changes)],
            )
            await asyncio.to_thread(
                self.index.remove, [(key, version) for key, _, version, _ in removals]
            )
            if self.embed_inputs and changes:
                for entry in await asyncio.to_thread(self.write_embed_inputs, changes):
                    await self.queue.push_embed_input(entry)
        await asyncio.to_thread(
            self.index.touch,
            [(record_key(record), record["fetched_at"]) for record in unchanged],
        )
        await self.report_outcomes(
            changes, unchanged, failed, [url for _, url, _, _ in removals]
        )
        for job in finalized:
            await self.queue.ack(job["task_id"])
            for key in (job.get("key"), job.get("input_key")):
                if key:
                    with contextlib.suppress(Exception):
                        await self.temp.delete(key)
        log.info(
            "published %d tasks, %d pages new or changed, %d withdrawn",
            len(finalized),
            len(changes),
            len(removed),
        )
        return len(finalized)

    async def on_pool(self, call, *args):
        return await asyncio.get_running_loop().run_in_executor(self.pool, call, *args)

    def read_job(self, job: dict) -> tuple[list[dict], list[dict]]:
        if job.get("kind") == "embed":
            self.publish_vectors(job)
            return [], []
        return self.collect(job)

    def read_rows(self, job: dict) -> list[dict]:
        """Every published column, read in byte ranges; the HTML is never fetched."""
        try:
            found = self.temp.client.head_object(
                Bucket=self.temp.bucket, Key=self.temp.path(job["key"])
            )
        except ClientError as exc:
            if exc.response.get("Error", {}).get("Code") in MISSING:
                raise UploadGone("expired") from None
            raise
        if job.get("etag") and found["ETag"].strip('"') != job["etag"].strip('"'):
            raise UploadGone("changed")
        remote = RangeFile(
            int(found["ContentLength"]),
            lambda start, end: self.temp.read_range_now(job["key"], start, end),
        )
        rows = read_rows(remote, ROW_COLUMNS, len(set(job["urls"])))
        if rows is None:
            raise UploadGone("unreadable")
        return rows

    def collect(self, job: dict) -> tuple[list[dict], list[dict]]:
        """The job's publishable records, and the assigned URLs it could not publish."""
        rows = self.read_rows(job)
        window = publish_window(job)
        captured = datetime.now(timezone.utc)
        given = set(job["urls"])
        assigned = given - set(job.get("skip", ()))
        records, published = [], set()
        for row in rows:
            if not publishable(row, assigned):
                continue
            published.add(row["url"])
            records.append(
                build_record(
                    row,
                    job["task_id"],
                    job["miner"],
                    window,
                    captured,
                    job.get("validator", ""),
                    job.get("validators", ()),
                )
            )
        return records, _failed(job, sorted(given - published))

    def decide(self, chosen: dict[str, dict]) -> tuple[list[dict], list[dict]]:
        """New and changed pages, and pages seen again unchanged, against the version index."""
        changes, unchanged = [], []
        for key, record in chosen.items():
            version = record_version(record)
            current = self.index.current(key)
            if current is None:
                changes.append(_change(record, key, "new", "", version))
            elif current.version == version or (
                (current.fetched_at, current.version) >= (record["fetched_at"], version)
            ):
                unchanged.append(record)
            else:
                changes.append(
                    _change(record, key, "changed", current.content_sha1, version)
                )
        return changes, unchanged

    async def report_outcomes(
        self,
        changes: list[dict],
        unchanged: list[dict],
        failed: list[dict],
        withdrawn: list[str] = (),
    ) -> None:
        rows = [
            {
                "url": c["assigned_url"],
                "outcome": outcomes.PUBLISHED,
                "task_id": c["task_id"],
            }
            for c in changes
        ]
        rows += [
            {
                "url": r["assigned_url"],
                "outcome": outcomes.UNCHANGED,
                "task_id": r["task_id"],
            }
            for r in unchanged
        ]
        rows += [{**row, "outcome": outcomes.FAILED} for row in failed]
        # Taken back out, so the bot sends them again for an honest crawl.
        rows += [
            {"url": url, "outcome": outcomes.DROPPED, "task_id": ""}
            for url in withdrawn
        ]
        for row in rows:
            row["host"] = domain_of(row["url"])
        try:
            await outcomes.write(self.temp, self.queue.redis, rows)
        except Exception:
            log.exception("could not write %d outcomes", len(rows))

    def read_temp(self, key: str, etag: str | None = None) -> bytes:
        extra = {"IfMatch": etag} if etag else {}
        try:
            return self.temp.client.get_object(
                Bucket=self.temp.bucket, Key=self.temp.path(key), **extra
            )["Body"].read()
        except ClientError as exc:
            code = exc.response.get("Error", {}).get("Code")
            if code in MISSING:
                raise UploadGone("expired") from None
            if code in RACED:
                raise UploadGone("changed") from None
            raise

    async def snapshot_daily(self) -> None:
        """A copy of the version index a day, beside the pages it describes."""
        day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if day == self.snapshot_day:
            return
        self.snapshot_day = day
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "index.sqlite")
            try:
                await asyncio.to_thread(self.index.backup_to, path)
                with open(path, "rb") as handle:
                    body = handle.read()
                await asyncio.to_thread(
                    self.pages.client.put_object,
                    Bucket=self.pages.bucket,
                    Key=self.pages.path(f"index/snapshots/{day}.sqlite"),
                    Body=body,
                    ContentType="application/vnd.sqlite3",
                )
            except Exception:
                log.exception("could not snapshot the version index")

    def publish_vectors(self, job: dict) -> tuple[list[dict], str | None]:
        given = read_parquet(self.read_temp(job["input_key"]), INPUT_SCHEMA)
        upload = self.read_temp(job["key"], job.get("etag"))
        vectors = {
            row["text_id"]: row["vector"] for row in read_parquet(upload, OUTPUT_SCHEMA)
        }
        rows = [
            {**row, "model": job["model"], "vector": vectors[row["text_id"]]}
            for row in given
        ]
        self.pages.client.put_object(
            Bucket=self.pages.bucket,
            Key=self.pages.path(job["vectors_key"]),
            Body=write_parquet(rows, VECTORS_SCHEMA),
            ContentType=PARQUET,
        )
        return [], None

    def write_embed_inputs(self, changes: list[dict]) -> list[dict]:
        """Texts of the new or changed pages, one file per future embed task."""
        pages = [change for change in changes if change["text"]]
        entries = []
        for start in range(0, len(pages), EMBED_INPUT_PAGES):
            batch = pages[start : start + EMBED_INPUT_PAGES]
            rows = [row for change in batch for row in embed_texts(change)]
            body = write_parquet(rows, INPUT_SCHEMA)
            key = f"embed-inputs/dt={datetime.now(timezone.utc):%Y-%m-%d}/{uuid.uuid4().hex}.parquet"
            self.temp.client.put_object(
                Bucket=self.temp.bucket,
                Key=self.temp.path(key),
                Body=body,
                ContentType=PARQUET,
            )
            entries.append(
                {
                    "input_key": key,
                    "input_sha256": hashlib.sha256(body).hexdigest(),
                    "texts": len(rows),
                    "chars": sum(len(row["text"]) for row in rows),
                    "pages": [
                        {
                            "url": change["url"],
                            "host": change["domain"],
                            "page_key": change["key"],
                            "content_sha1": change["content_sha1"],
                        }
                        for change in batch
                    ],
                }
            )
        return entries

    def write_changes(self, changes: list[dict]) -> str:
        """Every new, changed or removed page of the batch with its full record: the permanent copy."""
        now = datetime.now(timezone.utc)
        key = f"changes/dt={now:%Y-%m-%d}/{uuid.uuid4().hex}.parquet"
        sink = io.BytesIO()
        pq.write_table(
            pa.Table.from_pylist(changes, schema=CHANGE_SCHEMA),
            sink,
            compression="zstd",
            row_group_size=CHANGE_ROW_GROUP,
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


def embed_texts(change: dict) -> list[dict]:
    """The page as the index holds it: its head, its full text, then every passage."""
    title, text = change["title"] or "", change["text"]
    units = [("head", doc_head(title, text)), ("full", doc_full(title, text))]
    units += [("chunk", chunk) for chunk in para_chunks(text)]
    seen: dict[str, int] = {}
    rows = []
    for kind, unit in units:
        index = seen[kind] = seen.get(kind, -1) + 1
        rows.append(
            {
                "text_id": f"{change['key']}#{kind}{index}",
                "page_key": change["key"],
                "url": change["url"],
                "content_sha1": change["content_sha1"],
                "kind": kind,
                "index": index,
                "text": unit,
            }
        )
    return rows


def _rank(record: dict) -> tuple:
    return (
        record["assigned_url"] == record["url"],
        record["fetched_at"],
        record_version(record),
    )


def _change(
    record: dict, key: str, kind: str, previous_sha1: str, version: str
) -> dict:
    return {
        "key": key,
        "kind": kind,
        "previous_content_sha1": previous_sha1,
        "published_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "version": version,
        **{name: record[name] for name in RECORD_COLUMNS},
    }


def _removal(key: str, url: str, previous_sha1: str) -> dict:
    """A page taken back: readers drop it from what they hold."""
    return {
        **{name: None for name in RECORD_COLUMNS},
        "key": key,
        "kind": "removed",
        "previous_content_sha1": previous_sha1,
        "published_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "url": url,
        "domain": domain_of(url),
        "assigned_url": url,
        "json_ld_types": [],
        "headings": [],
        "validators": [],
    }


def _indexed(change: dict, seq: int | None = None, row: int | None = None) -> dict:
    return {
        "change_seq": seq,
        "change_row": row,
        "key": change["key"],
        "url": change["url"],
        "version": change["version"],
        "fetched_at": change["fetched_at"],
        "task_id": change["task_id"],
        "content_sha1": change["content_sha1"],
    }


def _failed(job: dict, urls) -> list[dict]:
    return [{"url": url, "task_id": job["task_id"]} for url in urls]


async def _sleep(stop: asyncio.Event, seconds: float) -> None:
    with contextlib.suppress(asyncio.TimeoutError):
        await asyncio.wait_for(stop.wait(), seconds)
