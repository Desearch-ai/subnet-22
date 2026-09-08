"""The URL frontier: every URL we have seen, written as parquet.

url_id is not stored: it is sha256(url) and recomputing it costs microseconds, while storing it
costs more disk than the URL itself because a hash does not compress.

Postgres holds the crawl schedule; this holds the URLs. Rows are buffered per host bucket and
flushed as whole files, so the write path is sequential and there is no per-URL statement.
"""

from __future__ import annotations

import hashlib
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

BUCKETS = 256
FLUSH_ROWS = 50_000

SCHEMA = pa.schema(
    [
        ("host", pa.string()),
        ("url", pa.string()),
        ("sitemap_id", pa.int64()),
        ("lastmod", pa.timestamp("us", tz="UTC")),
        ("changefreq", pa.string()),
        ("date_precision", pa.string()),
        ("first_seen_at", pa.timestamp("us", tz="UTC")),
    ]
)


def url_id(url: str) -> str:
    return hashlib.sha256(url.encode("utf-8")).hexdigest()[:32]


def host_bucket(host: str) -> int:
    return (
        int(hashlib.blake2b(host.encode("utf-8"), digest_size=2).hexdigest(), 16)
        % BUCKETS
    )


class Frontier:
    """Buffers URL rows per host bucket and writes parquet files under `root`."""

    def __init__(self, root: Path, flush_rows: int = FLUSH_ROWS):
        self.root = Path(root)
        self.flush_rows = flush_rows
        self.date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        self._buffers: dict[int, list[dict]] = {}
        self._lock = threading.Lock()
        self.files_written = 0
        self.rows_written = 0

    def add(self, host: str, sitemap_id: int | None, urls) -> int:
        bucket = host_bucket(host)
        now = datetime.now(timezone.utc)
        rows = [
            {
                "host": host,
                "url": url,
                "sitemap_id": sitemap_id,
                "lastmod": lastmod,
                "changefreq": changefreq,
                "date_precision": precision,
                "first_seen_at": now,
            }
            for url, lastmod, precision, changefreq in urls
        ]
        if not rows:
            return 0
        with self._lock:
            buffer = self._buffers.setdefault(bucket, [])
            buffer.extend(rows)
            ready = buffer if len(buffer) >= self.flush_rows else None
            if ready is not None:
                self._buffers[bucket] = []
        if ready is not None:
            self._write(bucket, ready)
        return len(rows)

    def flush(self) -> None:
        with self._lock:
            pending = [(b, rows) for b, rows in self._buffers.items() if rows]
            self._buffers = {}
        for bucket, rows in pending:
            self._write(bucket, rows)

    def _write(self, bucket: int, rows: list[dict]) -> Path:
        directory = self.root / f"dt={self.date}" / f"bucket={bucket:03d}"
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"part-{uuid.uuid4().hex[:12]}.parquet"
        rows.sort(key=lambda r: r["url"])
        table = pa.Table.from_pylist(rows, schema=SCHEMA)
        pq.write_table(table, path, compression="zstd", row_group_size=100_000)
        with self._lock:
            self.files_written += 1
            self.rows_written += len(rows)
        return path

    def stats(self) -> dict:
        with self._lock:
            buffered = sum(len(rows) for rows in self._buffers.values())
        return {
            "files": self.files_written,
            "rows": self.rows_written,
            "buffered": buffered,
        }
