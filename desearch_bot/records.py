"""A domain's and a sitemap's crawl state, as the bucket store keeps it."""

from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING

from .buckets import sitemap_id
from .schedule import DEFAULT_INTERVAL, Trust
from .states import State
from .visit import Known, KnownSitemap, SitemapUpdate

if TYPE_CHECKING:
    from .loop import DomainWrite


def new_domain(
    rank: int | None,
    tld_group: str | None,
    categories: Iterable[str] = (),
    state: State = State.NEW,
    reason: str | None = None,
    due: datetime | None = None,
) -> dict:
    """A domain the process has not visited yet."""
    return {
        "state": state.value,
        "reason": reason,
        "failures": 0,
        "due": _epoch(due),
        "rank": rank,
        "group": tld_group,
        "categories": sorted(categories),
        "urls": 0,
    }


def known(host: str, domain: dict, sitemaps: Iterable[tuple[str, dict]]) -> Known:
    """What a visit needs to know about a domain before it starts."""
    return Known(
        host=host,
        state=State(domain["state"]),
        failures=domain.get("failures", 0),
        last_ok_at=moment(domain.get("ok")),
        robots_checked_at=moment(domain.get("robots")),
        robots_allows=domain.get("allows"),
        crawl_delay=domain.get("delay"),
        language=domain.get("lang"),
        categories=frozenset(domain.get("categories") or ()),
        sitemaps={url: _known_sitemap(url, record) for url, record in sitemaps},
        canonical_host=domain.get("canonical"),
    )


def written_domain(previous: dict, write: DomainWrite, url_count: int) -> dict:
    """The domain record after a visit: robots fields only when robots was read."""
    visit = write.visit
    record = dict(previous)
    record.update(
        state=write.state.value,
        reason=write.reason,
        failures=write.failures,
        due=_epoch(write.next_due_at),
        ok=_epoch(write.last_ok_at),
        checked=_epoch(write.checked_at),
        canonical=write.canonical_host,
        urls=url_count,
    )
    if visit.robots_read:
        record.update(
            robots=_epoch(write.checked_at),
            robots_status=visit.robots_status,
            allows=visit.robots_allows,
            delay=visit.crawl_delay,
        )
    for key, value in (
        ("lang", visit.language),
        ("declared", visit.declared_lang),
        ("chars", visit.home_chars),
    ):
        if value is not None:
            record[key] = value
    return record


def written_sitemap(previous: dict | None, update: SitemapUpdate, at: datetime) -> dict:
    """The sitemap record after a read; its counts change only when its content did."""
    record = dict(previous or {})
    record.update(
        id=update.id,
        depth=update.depth,
        parent=update.parent_id,
        status=update.status,
        error=update.error,
        etag=update.etag,
        modified=update.last_modified,
        hash=update.content_hash,
        trust=update.trust.value,
        index_lastmod=update.index_lastmod,
        interval=int(update.interval.total_seconds()),
        next=_epoch(update.next_check_at),
        fetched=_epoch(at),
    )
    if update.kind:
        record["kind"] = update.kind
    if update.changed:
        record.update(
            urls=update.url_count, children=update.child_count, changed=_epoch(at)
        )
    record.setdefault("urls", 0)
    return record


def unread_sitemap(url: str, depth: int, parent: int | None) -> dict:
    """A sitemap found in an index that a later visit will read."""
    return {
        "id": sitemap_id(url),
        "depth": depth,
        "parent": parent,
        "status": "ok",
        "trust": Trust.UNKNOWN.value,
        "interval": int(DEFAULT_INTERVAL.total_seconds()),
        "next": None,
        "urls": 0,
    }


def url_count(sitemaps: Iterable[dict]) -> int:
    """URLs a domain lists across the sitemaps that last answered."""
    return sum(
        record.get("urls", 0) for record in sitemaps if record.get("status") == "ok"
    )


def _known_sitemap(url: str, record: dict) -> KnownSitemap:
    return KnownSitemap(
        record["id"],
        url,
        record.get("kind"),
        record.get("depth", 0),
        record.get("parent"),
        record.get("etag"),
        record.get("modified"),
        record.get("hash"),
        timedelta(seconds=record.get("interval", DEFAULT_INTERVAL.total_seconds())),
        moment(record.get("next")),
        Trust(record.get("trust", Trust.UNKNOWN.value)),
        record.get("index_lastmod"),
        record.get("urls", 0),
    )


def _epoch(value: datetime | None) -> int | None:
    return None if value is None else int(value.timestamp())


def moment(value: int | None) -> datetime | None:
    """A stored Unix timestamp as a UTC time."""
    return None if value is None else datetime.fromtimestamp(value, timezone.utc)
