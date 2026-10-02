"""What became of every URL, in order, for the bot that queued it."""

from __future__ import annotations

import io
import logging
import uuid
from datetime import datetime, timezone

import pyarrow as pa
import pyarrow.parquet as pq

PUBLISHED, UNCHANGED, FAILED, DROPPED = "published", "unchanged", "failed", "dropped"
SEQ = "outcomes:seq"
HOLES = "outcomes:holes"
LATEST_KEY = "outcomes/latest.json"
SCHEMA = pa.schema(
    [
        ("url", pa.string()),
        ("host", pa.string()),
        ("outcome", pa.string()),
        ("task_id", pa.string()),
        ("at", pa.timestamp("us", tz="UTC")),
    ]
)

log = logging.getLogger("task_api")


def seq_key(seq: int) -> str:
    return f"outcomes/seq/{seq:012d}.json"


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


async def number(storage, redis, key: str, rows: int) -> int:
    """Numbers a file already written; a number is only handed out for a file that exists."""
    at = int(await redis.incr(SEQ))
    try:
        await storage.put_json(seq_key(at), {"key": key, "rows": rows})
    except Exception:
        log.exception("could not index %s as %d", key, at)
        await redis.hset(HOLES, at, key)
        return at
    try:
        await storage.put_json(LATEST_KEY, {"seq": at}, cache_control="no-store")
    except Exception:
        log.warning("could not note %d as the newest outcome file", at)
    return at


async def fill_holes(storage, redis) -> int:
    """A number whose index write failed still gets one, so readers never wait on it."""
    filled = 0
    for at, key in (await redis.hgetall(HOLES)).items():
        try:
            await storage.put_json(seq_key(int(at)), {"key": key, "rows": None})
        except Exception:
            continue
        await redis.hdel(HOLES, at)
        filled += 1
    return filled


def rows_for(urls: list[str], outcome: str, task_id: str) -> list[dict]:
    from app.canonical import domain_of

    return [
        {"url": url, "host": domain_of(url), "outcome": outcome, "task_id": task_id}
        for url in urls
    ]
