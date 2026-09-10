"""Parse sitemap files into the addresses and dates they list."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timezone

ENTRY = re.compile(rb"<(url|sitemap)\b(.*?)</\1>", re.I | re.S)
LOC = re.compile(rb"<loc>\s*([^<\s]+)\s*</loc>", re.I)
LASTMOD = re.compile(rb"<lastmod>\s*([^<\s]+)\s*</lastmod>", re.I)
CHANGEFREQ = re.compile(rb"<changefreq>\s*(\w+)\s*</changefreq>", re.I)
SITEMAP_INDEX = re.compile(rb"<sitemapindex", re.I)
PUBLISHED = re.compile(
    rb"<news:publication_date>\s*([^<\s]+)\s*</news:publication_date>", re.I
)
NEWS_NAMESPACE = re.compile(rb"sitemap-news/0\.9", re.I)


@dataclass
class Entry:
    url: str
    lastmod: str | None = None
    changefreq: str | None = None
    published: str | None = None


def parse_entries(body: bytes) -> tuple[str, list[Entry]]:
    """Return the file kind and its entries, pairing each location with its own lastmod."""
    entries: list[Entry] = []
    kind = None
    for match in ENTRY.finditer(body):
        if kind is None:
            kind = "index" if match.group(1).lower() == b"sitemap" else "urlset"
        location = LOC.search(match.group(2))
        if not location:
            continue
        lastmod = LASTMOD.search(match.group(2))
        changefreq = CHANGEFREQ.search(match.group(2))
        published = PUBLISHED.search(match.group(2))
        entries.append(
            Entry(
                location.group(1).decode("utf-8", "replace"),
                lastmod.group(1).decode("utf-8", "replace") if lastmod else None,
                changefreq.group(1).decode("ascii", "replace").lower()
                if changefreq
                else None,
                published.group(1).decode("utf-8", "replace") if published else None,
            )
        )
    if entries:
        return kind, entries

    locations = LOC.findall(body)
    if not locations:
        return "invalid", []
    kind = "index" if SITEMAP_INDEX.search(body[:4096]) else "urlset"
    return kind, [Entry(loc.decode("utf-8", "replace")) for loc in locations]


def is_news(body: bytes) -> bool:
    """Whether the file uses Google's news sitemap format, which lists only recent articles."""
    return bool(NEWS_NAMESPACE.search(body[:4096]))


def has_time(value: str | None) -> bool:
    return bool(value) and "T" in value


def parse_lastmod(value: str | None) -> datetime | None:
    if not value:
        return None
    text = value.strip().replace("Z", "+00:00")
    for candidate in (text, text[:19], text[:10]):
        try:
            parsed = datetime.fromisoformat(candidate)
        except ValueError:
            continue
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
    return None
