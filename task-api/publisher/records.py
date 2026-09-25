from __future__ import annotations

import hashlib
import json
import uuid
from datetime import UTC, datetime, timedelta

import zstandard
from app.canonical import canonicalize, domain_of, url_sha1

PREFIX = "pages"
SOURCE = "subnet22"
CLOCK_SKEW = timedelta(minutes=5)
DEFAULT_CLAIM_S = 900
ROW_COLUMNS = [
    "url",
    "final_url",
    "status",
    "error",
    "fetched_at",
    "page_type",
    "title",
    "description",
    "lang",
    "canonical",
    "published",
    "author",
    "json_ld_types",
    "headings",
    "text",
    "text_sha256",
]
# Only content changes make a new version.
VERSIONED = (
    "url",
    "title",
    "published",
    "author",
    "lang",
    "text",
    "canonical",
    "page_type",
    "description",
    "json_ld_types",
    "headings",
)


def publish_window(job: dict) -> tuple[datetime, datetime]:
    """A row's fetch time must fall between claim and completion."""
    completed = job.get("completed_at")
    latest = (
        datetime.fromtimestamp(float(completed), UTC)
        if completed
        else datetime.now(UTC)
    )
    claim = timedelta(seconds=float(job.get("claim_ttl") or DEFAULT_CLAIM_S))
    return latest - claim - CLOCK_SKEW, latest


def build_record(
    row: dict,
    task_id: str,
    miner: str,
    window: tuple[datetime, datetime],
    captured_at: datetime | None = None,
    validator: str = "",
    validators: list[str] = (),
) -> dict:
    url = canonicalize(row["url"])
    text = row["text"] or ""
    earliest, latest = window
    # Clamped so a miner's clock cannot pick the winning version.
    fetched = min(max(_utc(row["fetched_at"], latest), earliest), latest)
    return {
        "url": url,
        "domain": domain_of(url),
        "title": row["title"] or "",
        "published": row["published"] or "",
        "author": row["author"] or "",
        "lang": row["lang"] or "",
        "text": text,
        "html": "",
        "fetched_at": _iso(fetched),
        "lastmod": "",
        "etag": "",
        "content_sha1": hashlib.sha1(text.encode()).hexdigest(),
        "source": SOURCE,
        "captured_at": _iso(captured_at or datetime.now(UTC)),
        "doc_id": str(uuid.uuid5(uuid.NAMESPACE_URL, url)),
        "assigned_url": row["url"],
        "final_url": row["final_url"],
        "canonical": row["canonical"],
        "status": row["status"],
        "page_type": row["page_type"],
        "description": row["description"],
        "json_ld_types": list(row["json_ld_types"] or []),
        "headings": list(row["headings"] or []),
        "text_sha256": row["text_sha256"],
        "task_id": task_id,
        "miner": miner,
        "validator": validator,
        "validators": list(validators) or ([validator] if validator else []),
    }


def page_key(url: str) -> str:
    return f"{PREFIX}/{domain_of(url)}/{url_sha1(url)}"


def record_key(record: dict) -> str:
    return page_key(record["url"])


def record_version(record: dict) -> str:
    stable = {name: record[name] for name in VERSIONED}
    return hashlib.sha1(
        json.dumps(stable, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()


def to_zstd(record: dict) -> bytes:
    # Writes the content size, which the engine's decoder needs.
    return zstandard.compress(json.dumps(record, ensure_ascii=False).encode(), 6)


def from_zstd(blob: bytes) -> dict:
    return json.loads(zstandard.decompress(blob))


def _utc(value, fallback: datetime) -> datetime:
    if not isinstance(value, datetime):
        return fallback
    return value if value.tzinfo else value.replace(tzinfo=UTC)


def _iso(value) -> str:
    if isinstance(value, datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=UTC)
        return value.astimezone(UTC).isoformat(timespec="seconds")
    return str(value or "")
