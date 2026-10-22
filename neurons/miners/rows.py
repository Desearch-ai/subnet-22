from __future__ import annotations

import asyncio
import io
from collections import Counter
from dataclasses import asdict

import pyarrow as pa
import pyarrow.parquet as pq

from desearch.extraction import EXTRACTION_V1, Page, extract, looks_blocked, v1
from desearch.extraction.schema import PAGE_SCHEMA, PAGE_SCHEMA_V2, sha256_hex
from desearch.fetch import Fetched, decode_html, page_url

ROW_GROUP_ROWS = 50


def build_row(fetched: Fetched, max_bytes: int, extraction: str) -> dict:
    if fetched.body is None:
        return error_row(fetched, fetched.error or "other", extraction)

    html = decode_html(fetched.body, fetched.charset)
    page = extractor(extraction)(html, page_url(fetched.url, fetched.final_url))
    if looks_blocked(fetched.status, html, page.text):
        return error_row(fetched, "blocked", extraction)
    if fetched.error:
        return error_row(fetched, fetched.error, extraction)
    if not page.text.strip():
        return error_row(fetched, "empty", extraction)

    fields = page_fields(html, page)
    if fields["html_bytes"] > max_bytes:
        return error_row(fetched, "too_large", extraction)
    return {**fetch_fields(fetched, extraction), "error": None, **fields}


def extractor(extraction: str):
    return v1.extract if extraction == EXTRACTION_V1 else extract


def page_schema(extraction: str) -> pa.Schema:
    return PAGE_SCHEMA if extraction == EXTRACTION_V1 else PAGE_SCHEMA_V2


def page_fields(html: str, page) -> dict:
    raw = html.encode("utf-8", "replace")
    return {
        "html_bytes": len(raw),
        "html": raw,
        "html_sha256": sha256_hex(raw),
        **asdict(page),
        "text_sha256": sha256_hex(page.text),
    }


def error_row(fetched: Fetched, error: str, extraction: str) -> dict:
    empty = v1.Page(page_type="") if extraction == EXTRACTION_V1 else Page(page_type="")
    return {
        **fetch_fields(fetched, extraction),
        "error": error,
        "html_bytes": 0,
        "html": None,
        "html_sha256": "",
        **asdict(empty),
        "text_sha256": "",
    }


def fetch_fields(fetched: Fetched, extraction: str) -> dict:
    fields = {
        "url": fetched.url,
        "final_url": fetched.final_url,
        "status": fetched.status,
        "fetched_at": fetched.fetched_at,
        "elapsed_ms": fetched.elapsed_ms,
        "content_type": fetched.content_type,
    }
    if extraction == EXTRACTION_V1:
        return fields
    return {**fields, "extraction": extraction}


class UploadWriter:
    """Rows go into the Parquet upload as they arrive, so a task never sits whole in memory."""

    def __init__(self, task_id: str, hotkey: str, extraction: str):
        schema = page_schema(extraction)
        metadata = {
            **(schema.metadata or {}),
            b"task_id": task_id.encode(),
            b"hotkey": hotkey.encode(),
        }
        self.extraction = extraction
        self.schema = schema.with_metadata(metadata)
        self.sink = io.BytesIO()
        self.writer = pq.ParquetWriter(self.sink, self.schema, compression="zstd")
        self.pending: list[dict] = []
        self.lock = asyncio.Lock()
        self.rows = 0
        self.errors: Counter[str] = Counter()

    @property
    def ok(self) -> int:
        return self.rows - sum(self.errors.values())

    async def add(self, row: dict) -> None:
        self.rows += 1
        if row["error"]:
            self.errors[row["error"]] += 1
        self.pending.append(row)
        if len(self.pending) >= ROW_GROUP_ROWS:
            group, self.pending = self.pending, []
            async with self.lock:
                await asyncio.to_thread(self.write_group, group)

    async def finish(self) -> bytes:
        async with self.lock:
            return await asyncio.to_thread(self.close)

    def write_group(self, rows: list[dict]) -> None:
        self.writer.write_table(pa.Table.from_pylist(rows, schema=self.schema))

    def close(self) -> bytes:
        if self.pending:
            self.write_group(self.pending)
            self.pending = []
        self.writer.close()
        return self.sink.getvalue()


def write_parquet(
    rows: list[dict], task_id: str, hotkey: str, extraction: str
) -> bytes:
    upload = UploadWriter(task_id, hotkey, extraction)
    for start in range(0, len(rows), ROW_GROUP_ROWS):
        upload.write_group(rows[start : start + ROW_GROUP_ROWS])
    return upload.close()
