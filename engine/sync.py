"""Turns vectors verified on the subnet into live segments the search service loads without a restart."""

from __future__ import annotations

import io
import json
import logging
import os
import shutil
import sqlite3
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import zstandard

from .build import LIVE, OUT, STORE, CorpusStats, write_index
from .chunking import para_chunks

log = logging.getLogger("engine.sync")

MODEL = os.environ.get("ENGINE_EMBED_MODEL", "qwen3-embedding-8b")
INTERVAL_S = 60.0
LOOKBACK_DAYS = 3
SEGMENT_TEXTS = 50_000
FETCH_WORKERS = 32
KIND_ORDER = {"head": 0, "full": 1, "chunk": 2}


class Bucket:
    """The pages bucket on R2, read-only."""

    def __init__(self):
        import boto3

        self.name = os.environ.get("CF_R2_PAGES_BUCKET", "desearch-pages")
        self.prefix = os.environ.get("CF_R2_PAGES_PREFIX", "")
        self.client = boto3.client(
            "s3",
            endpoint_url=os.environ["CF_R2_ENDPOINT"],
            aws_access_key_id=os.environ["CF_R2_ACCESS_KEY_ID"],
            aws_secret_access_key=os.environ["CF_R2_SECRET_ACCESS_KEY"],
            region_name="auto",
        )

    def keys(self, prefix: str) -> list[str]:
        found = []
        pages = self.client.get_paginator("list_objects_v2").paginate(
            Bucket=self.name, Prefix=self.prefix + prefix
        )
        for page in pages:
            found += [
                item["Key"].removeprefix(self.prefix)
                for item in page.get("Contents", [])
            ]
        return found

    def get(self, key: str) -> bytes | None:
        try:
            return self.client.get_object(Bucket=self.name, Key=self.prefix + key)[
                "Body"
            ].read()
        except self.client.exceptions.NoSuchKey:
            return None


class Synced:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path)
        self.db.execute(
            "CREATE TABLE IF NOT EXISTS synced (key TEXT PRIMARY KEY, segment TEXT, at REAL)"
        )
        self.db.commit()

    def new(self, keys: list[str]) -> list[str]:
        done = {row[0] for row in self.db.execute("SELECT key FROM synced")}
        return sorted(key for key in keys if key not in done)

    def close(self) -> None:
        self.db.close()

    def mark(self, keys: list[str], segment: str) -> None:
        self.db.executemany(
            "INSERT OR REPLACE INTO synced VALUES (?, ?, ?)",
            [(key, segment, time.time()) for key in keys],
        )
        self.db.commit()


def vector_keys(bucket, model: str, days: int = LOOKBACK_DAYS) -> list[str]:
    today = datetime.now(UTC).date()
    keys = []
    for back in range(days, -1, -1):
        day = today - timedelta(days=back)
        keys += bucket.keys(f"vectors/model={model}/dt={day.isoformat()}/")
    return keys


def pages_of(body: bytes) -> dict[str, dict]:
    """One vectors file as pages: content hash and each text's vector, keyed by page."""
    table = pq.read_table(io.BytesIO(body))
    pages: dict[str, dict] = {}
    for row in table.to_pylist():
        page = pages.setdefault(
            row["page_key"],
            {"content_sha1": row["content_sha1"], "url": row["url"], "rows": []},
        )
        page["rows"].append((KIND_ORDER[row["kind"]], row["index"], row["vector"]))
    return pages


def record(body: bytes | None) -> dict | None:
    if body is None:
        return None
    return json.loads(zstandard.ZstdDecompressor().decompress(body))


def indexable(page: dict, current: dict | None) -> np.ndarray | None:
    """The page's vectors in index order, if they belong to the text the page holds now."""
    if current is None or current.get("content_sha1") != page["content_sha1"]:
        return None
    rows = sorted(page["rows"], key=lambda row: row[:2])
    wanted = [
        (0, 0),
        (1, 0),
        *((2, i) for i in range(len(para_chunks(current["text"])))),
    ]
    if [row[:2] for row in rows] != wanted:
        return None
    return np.stack([np.frombuffer(row[2], dtype="<f2") for row in rows])


def build_segment(
    pages: list[tuple[dict, np.ndarray]], live: Path, store: Path, main: Path
) -> str:
    name = f"{int(time.time()):010d}-{uuid.uuid4().hex[:6]}"
    store.mkdir(parents=True, exist_ok=True)
    shard = store / f"{name}.npy"
    staging = shard.with_suffix(".tmp.npy")
    np.save(staging, np.concatenate([vectors for _, vectors in pages]))
    staging.rename(shard)

    def docs():
        pos = 0
        for current, vectors in pages:
            yield current, [(0, pos + i) for i in range(len(vectors))]
            pos += len(vectors)

    building = live / f"{name}.tmp"
    corpus = CorpusStats.of(main) if (main / "bm_vocab.npy").exists() else None
    write_index(building, [("subnet", docs())], [str(shard)], corpus)
    building.rename(live / name)
    (live / name / "READY").touch()
    return name


def sync_once(
    bucket,
    synced: Synced,
    model: str = MODEL,
    live: Path = LIVE,
    store: Path = STORE,
    main: Path = OUT,
) -> str | None:
    waiting = synced.new(vector_keys(bucket, model))
    if not waiting:
        return None
    taken, texts, pages = [], 0, {}
    for key in waiting:
        body = bucket.get(key)
        if body is None:
            continue
        found = pages_of(body)
        taken.append(key)
        pages.update(found)
        texts += sum(len(page["rows"]) for page in found.values())
        if texts >= SEGMENT_TEXTS:
            break

    with ThreadPoolExecutor(FETCH_WORKERS) as pool:
        currents = dict(zip(pages, pool.map(lambda k: record(bucket.get(k)), pages)))
    ready = []
    for key, page in pages.items():
        vectors = indexable(page, currents[key])
        if vectors is not None:
            ready.append((currents[key], vectors))
    name = build_segment(ready, live, store, main) if ready else ""
    synced.mark(taken, name)
    log.info(
        "synced %d vector files: %d of %d pages indexed%s",
        len(taken),
        len(ready),
        len(pages),
        f" into {name}" if name else "",
    )
    return name


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s"
    )
    bucket, synced = Bucket(), Synced(LIVE / "sync.db")
    for leftover in LIVE.glob("*.tmp"):
        shutil.rmtree(leftover)
    while True:
        try:
            if sync_once(bucket, synced) is not None:
                continue
        except Exception:
            log.exception("sync pass failed")
        time.sleep(INTERVAL_S)


if __name__ == "__main__":
    main()
