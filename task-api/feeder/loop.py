from __future__ import annotations

import asyncio
import json
import sqlite3
import sys
import time

from app.canonical import url_sha1

from desearch.client import TaskApiClient, TaskApiError

BATCH = 500
BATCHES_PER_REQUEST = 10
QUEUE_CAP = 20000
LOW_WATER = 1000
REFRESH_S = 86400.0
WAIT_S = 15.0
LOOKUP_CHUNK = 500


class SentUrls:
    def __init__(self, path: str):
        self.db = sqlite3.connect(path, check_same_thread=False)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=NORMAL")
        self.db.execute(
            "CREATE TABLE IF NOT EXISTS sent (id TEXT PRIMARY KEY, at REAL NOT NULL, lastmod REAL)"
        )
        columns = {row[1] for row in self.db.execute("PRAGMA table_info(sent)")}
        if "lastmod" not in columns:
            self.db.execute("ALTER TABLE sent ADD COLUMN lastmod REAL")
        self.db.commit()

    def due(self, rows: list[dict], refresh: float = REFRESH_S) -> list[dict]:
        now = time.time()
        ids = [url_sha1(row["url"]) for row in rows]
        known: dict[str, tuple[float, float | None]] = {}
        for start in range(0, len(ids), LOOKUP_CHUNK):
            chunk = ids[start : start + LOOKUP_CHUNK]
            marks = ",".join("?" * len(chunk))
            for id_, at, lastmod in self.db.execute(
                f"SELECT id, at, lastmod FROM sent WHERE id IN ({marks})", chunk
            ):
                known[id_] = (at, lastmod)
        keep = []
        for row, id_ in zip(rows, ids, strict=True):
            last = known.get(id_)
            if (
                last is None
                or last[0] <= now - refresh
                or _newer(row.get("lastmod"), last[1])
            ):
                keep.append(row)
        return keep

    def mark(self, rows: list[dict]) -> None:
        now = time.time()
        self.db.executemany(
            "INSERT OR REPLACE INTO sent VALUES (?, ?, ?)",
            [(url_sha1(row["url"]), now, row.get("lastmod")) for row in rows],
        )
        self.db.commit()

    def count(self) -> int:
        return self.db.execute("SELECT count(*) FROM sent").fetchone()[0]


def _newer(lastmod: float | None, sent: float | None) -> bool:
    return lastmod is not None and (sent is None or lastmod > sent)


async def queue_depth(client: TaskApiClient) -> int:
    health = await client.get("/v1/health")
    return int(health.get("queue_depth", 0))


async def enqueue(
    client: TaskApiClient, rows: list[dict], batch_target: int, sent_urls: SentUrls
) -> int:
    """A URL counts as sent only once the API took its batch."""
    sent = 0
    size = max(BATCH, batch_target * BATCHES_PER_REQUEST)
    for start in range(0, len(rows), size):
        chunk = rows[start : start + size]
        urls = [{"host": row["host"], "url": row["url"]} for row in chunk]
        try:
            await client.post(
                "/v1/admin/enqueue", {"urls": urls, "batch_target": batch_target}
            )
        except TaskApiError as exc:
            print(f"enqueue refused: {exc}", file=sys.stderr)
            break
        await asyncio.to_thread(sent_urls.mark, chunk)
        sent += len(chunk)
    return sent


async def feed_once(args, source, sent_urls: SentUrls, client: TaskApiClient) -> int:
    depth = await queue_depth(client)
    if depth > args.queue_cap:
        print(f"queue at {depth}, waiting", flush=True)
        return 0
    rows = await source()
    fresh = await asyncio.to_thread(sent_urls.due, rows, args.refresh)
    sent = await enqueue(client, fresh, args.batch_target, sent_urls) if fresh else 0
    print(
        f"{len(rows)} urls read, {len(fresh)} due, {sent} enqueued, queue was {depth}",
        flush=True,
    )
    return sent


async def wait_for_next_cycle(
    args, client: TaskApiClient, started: float, sent: int
) -> None:
    while True:
        left = args.interval - (time.monotonic() - started)
        if left <= 0:
            return
        if sent and await queue_depth(client) < args.low_water:
            return
        await asyncio.sleep(min(WAIT_S, left))


def load_domains(path: str, limit: int) -> list[str]:
    with open(path) as handle:
        listed = json.load(handle)
    return [entry["host"] if isinstance(entry, dict) else entry for entry in listed][
        :limit
    ]


async def run(args, source) -> int:
    sent_urls = SentUrls(args.state)
    print(f"{await asyncio.to_thread(sent_urls.count)} urls already sent", flush=True)

    async with TaskApiClient(args.api, args.key_uri) as client:
        while True:
            started = time.monotonic()
            sent = 0
            try:
                sent = await feed_once(args, source, sent_urls, client)
            except Exception as exc:
                print(
                    f"cycle failed: {type(exc).__name__}: {exc}",
                    file=sys.stderr,
                    flush=True,
                )
            if args.once:
                return 0
            await wait_for_next_cycle(args, client, started, sent)
