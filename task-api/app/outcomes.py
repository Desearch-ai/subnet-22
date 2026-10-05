"""What became of every URL, in order, for the bot that queued it."""

from __future__ import annotations

import io
import uuid
from datetime import datetime, timezone

import pyarrow as pa
import pyarrow.parquet as pq

from .feeds import Feed

PUBLISHED, UNCHANGED, FAILED, DROPPED = "published", "unchanged", "failed", "dropped"
FEED = Feed("outcomes")
SEQ, HOLES, LATEST_KEY = FEED.counter, FEED.holes, FEED.latest_key
seq_key, number, fill_holes = FEED.seq_key, FEED.number, FEED.fill_holes
SCHEMA = pa.schema(
    [
        ("url", pa.string()),
        ("host", pa.string()),
        ("outcome", pa.string()),
        ("task_id", pa.string()),
        ("at", pa.timestamp("us", tz="UTC")),
    ]
)


def file_key() -> str:
    day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    return f"outcomes/dt={day}/{uuid.uuid4().hex}.parquet"


def encode(rows: list[dict]) -> bytes:
    at = datetime.now(timezone.utc)
    table = pa.Table.from_pylist(
        [{**row, "at": row.get("at", at)} for row in rows], schema=SCHEMA
    )
    sink = io.BytesIO()
    pq.write_table(table, sink, compression="zstd")
    return sink.getvalue()


async def write(storage, redis, rows: list[dict]) -> int | None:
    if not rows:
        return None
    key = file_key()
    await storage.put_bytes(key, encode(rows), "application/vnd.apache.parquet")
    return await number(storage, redis, key, len(rows))


def rows_for(urls: list[str], outcome: str, task_id: str) -> list[dict]:
    from app.canonical import domain_of

    return [
        {"url": url, "host": domain_of(url), "outcome": outcome, "task_id": task_id}
        for url in urls
    ]
