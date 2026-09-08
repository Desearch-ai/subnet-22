"""Walk a sitemap tree and collect the page URLs it lists.

Index files are followed breadth-first up to a depth limit. Collection stops at a URL limit so one
large publisher cannot consume a whole run; children that were not reached stay in the database
with status `pending` and can be walked later.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timezone

ENTRY = re.compile(rb"<(url|sitemap)\b(.*?)</\1>", re.I | re.S)
LOC = re.compile(rb"<loc>\s*([^<\s]+)\s*</loc>", re.I)
LASTMOD = re.compile(rb"<lastmod>\s*([^<\s]+)\s*</lastmod>", re.I)
CHANGEFREQ = re.compile(rb"<changefreq>\s*(\w+)\s*</changefreq>", re.I)
SITEMAP_INDEX = re.compile(rb"<sitemapindex", re.I)
DATE_IN_PATH = re.compile(r"/(\d{4})[-/](\d{2})(?:[-/](\d{2}))?/")
YEAR_IN_NAME = re.compile(r"(?:^|\D)(19\d{2}|20\d{2})(?:\D|$)")


@dataclass
class Entry:
    url: str
    lastmod: str | None = None
    changefreq: str | None = None


@dataclass
class Walk:
    urls: list[tuple[str, datetime | None, str, str | None]] = field(default_factory=list)
    children: list[Entry] = field(default_factory=list)
    truncated: bool = False


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
        entries.append(
            Entry(
                location.group(1).decode("utf-8", "replace"),
                lastmod.group(1).decode("utf-8", "replace") if lastmod else None,
                changefreq.group(1).decode("ascii", "replace").lower() if changefreq else None,
            )
        )
    if entries:
        return kind, entries

    locations = LOC.findall(body)
    if not locations:
        return "invalid", []
    kind = "index" if SITEMAP_INDEX.search(body[:4096]) else "urlset"
    return kind, [Entry(loc.decode("utf-8", "replace")) for loc in locations]


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


def date_precision(url: str, lastmod: str | None) -> str:
    if lastmod:
        return "day"
    match = DATE_IN_PATH.search(url)
    if match:
        return "day" if match.group(3) else "month"
    return "year" if YEAR_IN_NAME.search(url.rsplit("/", 1)[-1]) else "none"


def collect(kind: str, entries: list[Entry], remaining: int) -> Walk:
    walk = Walk()
    if kind == "index":
        walk.children = entries
        return walk
    for entry in entries:
        if len(walk.urls) >= remaining:
            walk.truncated = True
            break
        walk.urls.append(
            (
                entry.url,
                parse_lastmod(entry.lastmod),
                date_precision(entry.url, entry.lastmod),
                entry.changefreq,
            )
        )
    return walk
