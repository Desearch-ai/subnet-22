"""Every URL we know: normalised once, stored once, filed by domain."""

from __future__ import annotations

import re
import string
import struct
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote, urlsplit

from rocksdict import DBCompressionType, Options, Rdict, WriteBatch

UNRESERVED = frozenset(string.ascii_letters + string.digits + "-._~")
PERCENT = re.compile(r"%([0-9A-Fa-f]{2})")
SAFE_PATH = "/:@!$&'()*+,;=-._~%"
SAFE_QUERY = SAFE_PATH + "?"
DEFAULT_PORTS = {"http": 80, "https": 443}
TRACKING = frozenset({"gclid", "fbclid", "msclkid"})

HTTPS = 1
WWW = 2
TIMED = 4

RECORD = struct.Struct("<qIIIIIB")


@dataclass(frozen=True)
class Url:
    key: bytes
    flags: int

    def fetchable(self) -> str:
        """The address a miner requests, in the scheme and form the site listed it."""
        rest = self.key.split(b"\x00", 1)[1].decode()
        scheme = "https" if self.flags & HTTPS else "http"
        return f"{scheme}://{'www.' if self.flags & WWW else ''}{rest}"


@dataclass
class Record:
    sitemap_id: int
    lastmod: int
    first_seen: int
    last_seen: int
    crawled_at: int = 0
    pushed_at: int = 0
    flags: int = 0

    def pack(self) -> bytes:
        return RECORD.pack(
            self.sitemap_id,
            self.lastmod,
            self.first_seen,
            self.last_seen,
            self.crawled_at,
            self.pushed_at,
            self.flags,
        )

    @classmethod
    def unpack(cls, data: bytes) -> Record:
        return cls(*RECORD.unpack(data))


@dataclass(frozen=True)
class Listing:
    listed: int
    new: int
    moved: int


def parse(url: str, domain: str) -> Url | None:
    """Normalise a sitemap URL, or None when it is not a web page on this domain."""
    try:
        domain = _ascii(domain)
        parts = urlsplit(url.strip())
        scheme = parts.scheme.lower()
        if scheme not in DEFAULT_PORTS or not parts.hostname:
            return None
        host = _ascii(parts.hostname)
        port = parts.port
    except (ValueError, UnicodeError):
        return None
    if host != domain and not host.endswith("." + domain):
        return None
    bare = host.removeprefix("www.")
    netloc = bare if port in (None, DEFAULT_PORTS[scheme]) else f"{bare}:{port}"
    query = _query(parts.query)
    rest = (
        netloc + (_tidy(parts.path, SAFE_PATH) or "/") + (f"?{query}" if query else "")
    )
    flags = (HTTPS if scheme == "https" else 0) | (WWW if host != bare else 0)
    return Url(f"{domain}\x00{rest}".encode(), flags)


class UrlStore:
    """Every URL we know, filed by domain so a domain's pages sit together on disk."""

    def __init__(self, path: Path):
        options = Options(raw_mode=True)
        options.create_if_missing(True)
        options.set_compression_type(DBCompressionType.zstd())
        self.db = Rdict(str(path), options)

    def __enter__(self) -> UrlStore:
        return self

    def __exit__(self, *_) -> None:
        self.close()

    def close(self) -> None:
        self.db.close()

    def record_listing(
        self, sitemap_id: int, entries: list[tuple[Url, int, bool]], now: int
    ) -> Listing:
        """Store what one sitemap lists right now; each URL it names becomes its own."""
        unique: dict[bytes, tuple[Url, int, bool]] = {}
        for url, lastmod, timed in entries:
            unique.setdefault(url.key, (url, lastmod, timed))
        keys = list(unique)
        if not keys:
            return Listing(0, 0, 0)

        batch = WriteBatch(raw_mode=True)
        new = moved = 0
        for key, raw in zip(keys, self.db.get(keys)):
            url, lastmod, timed = unique[key]
            if raw is None:
                record = Record(
                    sitemap_id, lastmod, now, now, flags=url.flags | _timed(timed)
                )
                new += 1
            else:
                record = Record.unpack(raw)
                if lastmod and lastmod != record.lastmod:
                    record.lastmod = lastmod
                    record.flags = (record.flags & ~TIMED) | _timed(timed)
                    moved += 1
                record.sitemap_id = sitemap_id
                record.last_seen = now
            batch.put(key, record.pack())
        self.db.write(batch)
        return Listing(len(keys), new, moved)

    def get(self, url: Url) -> Record | None:
        raw = self.db.get(url.key)
        return None if raw is None else Record.unpack(raw)

    def domain(self, domain: str) -> Iterator[tuple[Url, Record]]:
        """Every URL stored for a domain, in key order."""
        prefix = f"{_ascii(domain)}\x00".encode()
        for key, raw in self.db.items(from_key=prefix):
            if not key.startswith(prefix):
                return
            record = Record.unpack(raw)
            yield Url(key, record.flags & (HTTPS | WWW)), record

    def estimate(self) -> int:
        return self.db.property_int_value("rocksdb.estimate-num-keys") or 0


def _ascii(host: str) -> str:
    return host.strip().rstrip(".").encode("idna").decode("ascii").lower()


def _timed(timed: bool) -> int:
    return TIMED if timed else 0


def _tidy(part: str, safe: str) -> str:
    return quote(PERCENT.sub(_unescape, part), safe=safe)


def _unescape(match: re.Match) -> str:
    char = chr(int(match.group(1), 16))
    return char if char in UNRESERVED else "%" + match.group(1).upper()


def _query(query: str) -> str:
    kept = []
    for pair in query.split("&"):
        name = pair.split("=", 1)[0].lower()
        if pair and name not in TRACKING and not name.startswith("utm_"):
            kept.append(_tidy(pair, SAFE_QUERY))
    return "&".join(sorted(kept))
