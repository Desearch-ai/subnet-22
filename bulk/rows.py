"""Upload-shaped rows and the folder of batch files `publisher bulk` publishes."""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from desearch.extraction import (
    EXTRACTION_V2,
    MAIN,
    Page,
    text_headings,
    without_surrogates,
)
from desearch.extraction.schema import PAGE_SCHEMA, sha256_hex
from desearch.extraction.text.dom import MAX_TEXT_CHARS
from desearch.extraction.text.lines import clean
from desearch.extraction.text.markdown import Line, finish

ROWS_PER_FILE = 5_000
ROW_GROUP_ROWS = 500
MANIFEST = "manifest.json"
REMOVED = "removed.txt"
MAX_TAGS = 20
# The planner keeps a paper's first 12 parts.
PAPER = "paper"


def upload_row(url: str, page: Page, fetched_at: datetime, content_type: str) -> dict:
    """A row as a miner's upload holds a fetched page, without the HTML."""
    return {
        "url": url,
        "final_url": url,
        "status": 200,
        "error": None,
        "fetched_at": fetched_at,
        "elapsed_ms": 0,
        "content_type": content_type,
        "html_bytes": 0,
        "html": None,
        "html_sha256": "",
        **asdict(page),
        "text_sha256": sha256_hex(page.text),
        "extraction": EXTRACTION_V2,
    }


def paper_page(
    title: str,
    lines: list[Line],
    *,
    lang: str,
    url: str,
    published: str = "",
    modified: str = "",
    author: str = "",
    site_name: str = "",
    tags: list[str] | None = None,
) -> Page:
    """A paper's page in v2's text conventions: the title line, then cleaned lines with `##` headings."""
    text = finish(title, clean(lines), MAX_TEXT_CHARS)
    return without_surrogates(
        Page(
            page_type=PAPER,
            text_mode=MAIN,
            title=title,
            lang=lang,
            canonical=url,
            published=published,
            modified=modified,
            author=author,
            site_name=site_name,
            tags=(tags or [])[:MAX_TAGS],
            headings=text_headings(text),
            text=text,
        )
    )


def authors_line(names: list[str], most: int = 10) -> str:
    names = [n for n in names if n]
    if len(names) > most:
        return ", ".join(names[:most]) + ", et al."
    return ", ".join(names)


def parse_as_of(text: str) -> datetime:
    moment = datetime.fromisoformat(text.replace("Z", "+00:00"))
    if moment.tzinfo is None:
        raise ValueError(
            f"--as-of {text!r} needs a time zone, as in 2026-11-01T00:00:00Z"
        )
    return moment.astimezone(timezone.utc)


class Batches:
    """Rows written to numbered parquet files in one folder, with a manifest naming the source, run and time."""

    def __init__(
        self,
        out: Path,
        source: str,
        run: str,
        as_of: datetime,
        rows_per_file: int = ROWS_PER_FILE,
    ):
        self.out = out
        self.source = source
        self.run = run
        self.as_of = as_of
        self.rows_per_file = rows_per_file
        self.files: list[str] = []
        self.removed: list[str] = []
        self.rows = 0
        self.pending: list[dict] = []
        self.writer: pq.ParquetWriter | None = None
        self.in_file = 0
        out.mkdir(parents=True, exist_ok=True)
        if (out / MANIFEST).exists():
            raise FileExistsError(
                f"{out} already holds a run; give each run its own folder"
            )

    def add(self, row: dict) -> None:
        if self.writer is None:
            name = f"{self.source}-{len(self.files) + 1:06d}.parquet"
            self.files.append(name)
            self.writer = pq.ParquetWriter(
                self.out / name, PAGE_SCHEMA, compression="zstd"
            )
        self.pending.append(row)
        self.in_file += 1
        self.rows += 1
        if len(self.pending) >= ROW_GROUP_ROWS:
            self.flush()
        if self.in_file >= self.rows_per_file:
            self.close_file()

    def remove(self, url: str) -> None:
        """A page the source deleted, taken down when the run is published."""
        self.removed.append(url)

    def flush(self) -> None:
        if self.pending and self.writer is not None:
            self.writer.write_table(
                pa.Table.from_pylist(self.pending, schema=PAGE_SCHEMA)
            )
            self.pending = []

    def close_file(self) -> None:
        self.flush()
        if self.writer is not None:
            self.writer.close()
        self.writer = None
        self.in_file = 0

    def close(self, counts: dict[str, int]) -> None:
        """Closes the last file and writes the manifest, which marks the run complete."""
        self.close_file()
        if self.removed:
            (self.out / REMOVED).write_text("".join(f"{url}\n" for url in self.removed))
        manifest = {
            "source": self.source,
            "run": self.run,
            "as_of": self.as_of.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "files": self.files,
            "removed": REMOVED if self.removed else None,
            "rows": self.rows,
            "counts": counts,
        }
        (self.out / MANIFEST).write_text(json.dumps(manifest, indent=1))
