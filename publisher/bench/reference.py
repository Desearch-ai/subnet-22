"""Runs the Python publisher on local uploads and dumps what it decided: reference.py SAMPLE_DIR WORK_DIR."""

from __future__ import annotations

import asyncio
import hashlib
import io
import json
import sqlite3
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from botocore.exceptions import ClientError

import publisher.worker as worker
from publisher.index import VersionIndex
from publisher.records import record_key, record_version

NOW = 1791311000.0
OLDER = "2026-10-01T00:00:00+00:00"
NEWER = "2026-12-01T00:00:00+00:00"


class Clock(datetime):
    """The worker's clock, held still so both publishers stamp the same times."""

    at = NOW

    @classmethod
    def now(cls, tz=None):
        return datetime.fromtimestamp(cls.at, tz)


class Head:
    def __init__(self, temp):
        self.temp = temp

    def head_object(self, Bucket, Key):
        path = self.temp.files.get(Key)
        if path is None or not path.exists():
            raise ClientError({"Error": {"Code": "NoSuchKey"}}, "HeadObject")
        return {"ContentLength": path.stat().st_size, "ETag": self.temp.etags[Key]}


class Temp:
    """Uploads read from local files through the ranged reads the R2 client serves."""

    bucket, prefix = "temp", ""

    def __init__(self, folder: Path, jobs: list[dict]):
        keyed = [job for job in jobs if job.get("key")]
        self.files = {job["key"]: folder / f"{job['task_id']}.parquet" for job in keyed}
        self.etags = {job["key"]: job.get("etag") or "" for job in keyed}
        self.client = Head(self)
        self.objects = {}
        self.bytes = self.requests = 0

    def path(self, key):
        return key

    def read_range_now(self, key, start, end):
        self.requests += 1
        self.bytes += end - start + 1
        with open(self.files[key], "rb") as f:
            f.seek(start)
            return f.read(end - start + 1)

    async def put_bytes(self, key, body, content_type=None):
        self.objects[key] = body

    async def put_json(self, key, value, cache_control=None):
        self.objects[key] = value

    async def delete(self, key):
        pass


class Pages:
    bucket, prefix = "pages", ""

    def __init__(self):
        self.objects = {}
        self.client = self

    def put_object(self, Bucket, Key, Body, ContentType=None):
        self.objects[Key] = Body

    def path(self, key):
        return key

    async def put_json(self, key, value, cache_control=None):
        self.objects[key] = value

    async def stat(self, key):
        return key in self.objects


class Redis:
    def __init__(self):
        self.values, self.hashes = {}, {}

    async def incr(self, key):
        self.values[key] = self.values.get(key, 0) + 1
        return self.values[key]

    async def hset(self, name, key, value):
        self.hashes.setdefault(name, {})[key] = value

    async def hgetall(self, name):
        return dict(self.hashes.get(name, {}))

    async def hdel(self, name, key):
        self.hashes.get(name, {}).pop(key, None)


class Queue:
    def __init__(self):
        self.redis = Redis()
        self.acked, self.lost = [], []

    async def extend_claim(self, task_id):
        pass

    async def mark_lost(self, task_id):
        self.lost.append(task_id)

    async def ack(self, task_id):
        self.acked.append(task_id)


def seed_for(records: list[dict]) -> list[dict]:
    """Half the pages already indexed: most at the same version, a tenth at another, older or newer."""
    seed, seen = [], set()
    for record in records:
        key = record_key(record)
        if key in seen:
            continue
        seen.add(key)
        bucket = int(hashlib.sha1(key.encode()).hexdigest()[:8], 16) % 20
        if bucket < 10:
            continue
        entry = {
            "key": key,
            "url": record["url"],
            "version": record_version(record),
            "fetched_at": record["fetched_at"] if bucket < 14 else OLDER,
            "task_id": "seeded",
            "content_sha1": record["content_sha1"],
            "change_seq": None,
            "change_row": None,
        }
        if bucket == 18:
            entry.update(version="0" * 40, fetched_at=OLDER, content_sha1="1" * 40)
        if bucket == 19:
            entry.update(version="f" * 40, fetched_at=NEWER, content_sha1="2" * 40)
        seed.append(entry)
    return seed


def recrawl(sample: Path, work: Path, job: dict) -> dict:
    """The same URLs crawled again a minute later, every other page changed and fetched 30 seconds later."""
    table = pq.read_table(sample / f"{job['task_id']}.parquet")
    rows = table.to_pylist()
    for row in rows[::2]:
        row["text"] = (row["text"] or "") + " (updated)"
        row["fetched_at"] = row["fetched_at"] + timedelta(seconds=30)
    task_id = job["task_id"] + "-recrawl"
    pq.write_table(
        pa.Table.from_pylist(rows, schema=table.schema),
        work / "uploads" / f"{task_id}.parquet",
        compression="zstd",
        row_group_size=50,
    )
    return {
        **job,
        "task_id": task_id,
        "key": job["key"] + "-recrawl",
        "completed_at": job["completed_at"] + 60,
    }


def batches_for(sample: Path, work: Path, jobs: list[dict]) -> list[list[dict]]:
    """The sample as one batch, then one that withdraws two tasks while one of them is re-crawled."""
    taken, replayed, recrawled = jobs[0], jobs[1], jobs[2]
    return [
        jobs,
        [
            {
                "task_id": "withdraw-1",
                "kind": "withdraw",
                "task_ids": [taken["task_id"], recrawled["task_id"]],
            },
            taken,
            replayed,
            recrawl(sample, work, recrawled),
            {**replayed, "task_id": "gone-1", "key": "submitted/gone-1.parquet"},
        ],
    ]


def index_rows(path: Path) -> dict:
    db = sqlite3.connect(path)
    names = [
        "url",
        "version",
        "fetched_at",
        "task_id",
        "content_sha1",
        "change_seq",
        "change_row",
    ]
    pages = {
        row[0]: dict(zip(names, row[1:]))
        for row in db.execute(f"SELECT key, {', '.join(names)} FROM pages")
    }
    withdrawn = sorted(row[0] for row in db.execute("SELECT task_id FROM withdrawn"))
    db.close()
    return {"pages": pages, "withdrawn": withdrawn}


def main() -> None:
    sample, work = Path(sys.argv[1]), Path(sys.argv[2])
    (work / "uploads").mkdir(parents=True, exist_ok=True)
    (work / "python-changes").mkdir(exist_ok=True)
    jobs = json.loads((sample / "jobs.json").read_text())
    for job in jobs:
        link = work / "uploads" / f"{job['task_id']}.parquet"
        if not link.exists():
            link.symlink_to(sample / f"{job['task_id']}.parquet")
    batches = batches_for(sample, work, jobs)
    worker.datetime = Clock
    temp = Temp(work / "uploads", [job for batch in batches for job in batch])

    seeding = worker.Publisher(Queue(), temp, Pages(), workers=8)
    seed = seed_for([record for job in jobs for record in seeding.collect(job)[0]])
    seeding.close()
    index_path = work / "python-index.sqlite"
    for suffix in ("", "-wal", "-shm"):
        Path(f"{index_path}{suffix}").unlink(missing_ok=True)
    index = VersionIndex(str(index_path))
    index.store(seed)

    queue, pages = Queue(), Pages()
    temp.bytes = temp.requests = 0
    publisher = worker.Publisher(queue, temp, pages, workers=8, index=index)
    calls, collected = {}, {}

    def recording(name, function):
        def wrapped(*args):
            result = function(*args)
            calls.setdefault(name, []).append((args, result))
            return result

        return wrapped

    collect = publisher.collect

    def collect_and_keep(job):
        try:
            records, failed = collect(job)
        except Exception as exc:
            collected[job["task_id"]] = {"error": type(exc).__name__}
            raise
        collected[job["task_id"]] = {
            "records": [
                {**r, "key": record_key(r), "version": record_version(r)}
                for r in records
            ],
            "failed": failed,
        }
        return records, failed

    publisher.collect = collect_and_keep
    publisher.decide = recording("decide", publisher.decide)
    for name in ("store", "touch", "remove", "of_tasks"):
        setattr(index, name, recording(name, getattr(index, name)))

    dump = {"batches": []}
    for number, batch in enumerate(batches, 1):
        Clock.at = NOW + 60 * (number - 1)
        calls.clear()
        collected.clear()
        pages.objects.clear()
        temp.objects.clear()
        reading = [job for job in batch if job.get("kind") != "withdraw"]

        async def run():
            await publisher.write_batch(
                batch,
                asyncio.create_task(publisher.read_all(reading)),
                time.monotonic(),
            )

        asyncio.run(run())
        changes, unchanged = calls["decide"][0][1]
        change_file = None
        for key, body in pages.objects.items():
            if key.startswith("changes/dt="):
                change_file = work / "python-changes" / f"{number:012d}.parquet"
                change_file.write_bytes(body)
        outcomes = []
        for key, body in temp.objects.items():
            if key.startswith("outcomes/dt="):
                outcomes += pq.read_table(io.BytesIO(body)).drop(["at"]).to_pylist()
        dump["batches"].append(
            {
                "now": Clock.at,
                "jobs": [
                    {"task_id": job["task_id"], **collected.get(job["task_id"], {})}
                    for job in reading
                ],
                "changes": changes,
                "unchanged": [
                    {
                        "key": record_key(r),
                        "version": record_version(r),
                        "fetched_at": r["fetched_at"],
                    }
                    for r in unchanged
                ],
                "stored": [
                    entry for args, _ in calls.get("store", []) for entry in args[0]
                ],
                "touched": [
                    list(pair) for args, _ in calls.get("touch", []) for pair in args[0]
                ],
                "removed": [
                    list(pair)
                    for args, _ in calls.get("remove", [])
                    for pair in args[0]
                ],
                "of_tasks": [
                    list(row)
                    for _, result in calls.get("of_tasks", [])
                    for row in result
                ],
                "change_file": str(change_file) if change_file else None,
                "outcomes": outcomes,
                "lost": list(queue.lost),
                "acked": list(queue.acked),
            }
        )
        queue.lost.clear()
        queue.acked.clear()
    publisher.close()
    dump["index"] = index_rows(index_path)
    dump["traffic"] = {"bytes": temp.bytes, "requests": temp.requests}
    (work / "batches.json").write_text(json.dumps(batches))
    (work / "seed.json").write_text(json.dumps(seed))
    (work / "python.json").write_text(json.dumps(dump, ensure_ascii=False))
    print(
        f"python: {sum(len(b['changes']) for b in dump['batches'])} changes in {len(batches)} batches,"
        f" {len(dump['index']['pages'])} pages indexed, {temp.bytes} bytes in {temp.requests} ranged reads"
    )


if __name__ == "__main__":
    main()
