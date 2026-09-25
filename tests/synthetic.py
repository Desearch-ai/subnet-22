from __future__ import annotations

import hashlib
import io
from datetime import datetime, timezone

import pyarrow as pa
import pyarrow.parquet as pq

from desearch.extraction import extract
from desearch.extraction.schema import PAGE_SCHEMA, sha256_hex
from neurons.validators.scoring import FetchedPage, sample_seed, score

SEED = sample_seed("task-1", "validator-hotkey")
VOCAB = (
    "river stone market garden window paper city harbor winter music signal forest"
    " bridge letter engine silver ocean mountain field lantern"
).split()
CHALLENGE = (
    "<html><head><title>Just a moment...</title></head><body>"
    "<div id=cf-browser-verification>Checking your browser before accessing the site.</div>"
    "</body></html>"
)


def synthetic_html(n: int, title: str | None = None) -> str:
    words = [
        VOCAB[b % len(VOCAB)] for b in hashlib.sha256(str(n).encode()).digest() * 8
    ]
    title = title or f"Page {n}"
    return (
        f"<html><head><title>{title}</title></head><body><article><h1>{title}</h1>"
        f"<p>{' '.join(words)}.</p></article></body></html>"
    )


def page_row(url: str, html: str, final_url: str = "", text: str | None = None) -> dict:
    final_url = final_url or url
    page = extract(html, final_url)
    raw = html.encode()
    text = page.text if text is None else text
    return {
        "url": url,
        "final_url": final_url,
        "status": 200,
        "error": None,
        "fetched_at": datetime.now(timezone.utc),
        "elapsed_ms": 120,
        "content_type": "text/html; charset=utf-8",
        "html_bytes": len(raw),
        "html": raw,
        "html_sha256": sha256_hex(raw),
        "page_type": page.page_type,
        "title": page.title,
        "description": page.description,
        "lang": page.lang,
        "canonical": page.canonical,
        "published": page.published,
        "author": page.author,
        "json_ld_types": page.json_ld_types,
        "headings": page.headings,
        "text": text,
        "text_sha256": sha256_hex(text),
    }


def error_row(url: str, error: str = "http_4xx", status: int = 404) -> dict:
    return {
        **page_row(url, ""),
        "status": status,
        "error": error,
        "content_type": "",
        "html_bytes": 0,
        "html": None,
        "html_sha256": "",
        "page_type": "",
        "text": "",
        "text_sha256": "",
    }


def synthetic_url(n: int) -> str:
    return f"https://site{n}.example/news/story-{n}"


def synthetic(count: int, errors: int = 0):
    urls = [synthetic_url(n) for n in range(count)]
    rows = [
        error_row(url) if n < errors else page_row(url, synthetic_html(n))
        for n, url in enumerate(urls)
    ]
    fetched = {url: FetchedPage(200, synthetic_html(n)) for n, url in enumerate(urls)}
    return rows, urls, fetched


def to_parquet(rows: list[dict], schema: pa.Schema = PAGE_SCHEMA, **metadata) -> bytes:
    schema = schema.with_metadata({k.encode(): v.encode() for k, v in metadata.items()})
    sink = io.BytesIO()
    pq.write_table(pa.Table.from_pylist(rows, schema=schema), sink, compression="zstd")
    return sink.getvalue()


def run(rows, assigned, fetched, min_samples=5, match_ratio=0.8) -> dict:
    return score(rows, assigned, fetched, SEED, min_samples, match_ratio)
