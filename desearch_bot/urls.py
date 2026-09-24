"""Every URL we know: normalised once, stored once, filed by domain."""

from __future__ import annotations

import re
import string
import struct
from dataclasses import dataclass
from functools import lru_cache
from urllib.parse import quote, urlsplit


UNRESERVED = frozenset(string.ascii_letters + string.digits + "-._~")
PERCENT = re.compile(r"%([0-9A-Fa-f]{2})")
SAFE_PATH = "/:@!$&'()*+,;=-._~%"
SAFE_QUERY = SAFE_PATH + "?"
DEFAULT_PORTS = {"http": 80, "https": 443}
TRACKING = frozenset({"gclid", "fbclid", "msclkid"})
# Text made only of these characters comes out of normalisation unchanged.
PLAIN = re.compile(r"[A-Za-z0-9\-._~/:@!$&'()*+,;=]*\Z")
SIMPLE_HOST = re.compile(r"[A-Za-z0-9.\-]+\Z")

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
        if scheme not in DEFAULT_PORTS:
            return None
        if SIMPLE_HOST.match(parts.netloc):
            host, port = _ascii(parts.netloc), None
        elif parts.hostname:
            host, port = _ascii(parts.hostname), parts.port
        else:
            return None
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


@lru_cache(maxsize=1 << 16)
def _ascii(host: str) -> str:
    return host.strip().rstrip(".").encode("idna").decode("ascii").lower()


def _tidy(part: str, safe: str) -> str:
    if PLAIN.match(part):
        return part
    return quote(PERCENT.sub(_unescape, part), safe=safe)


def _unescape(match: re.Match) -> str:
    char = chr(int(match.group(1), 16))
    return char if char in UNRESERVED else "%" + match.group(1).upper()


def _query(query: str) -> str:
    if not query:
        return ""
    kept = []
    for pair in query.split("&"):
        name = pair.split("=", 1)[0].lower()
        if pair and name not in TRACKING and not name.startswith("utm_"):
            kept.append(_tidy(pair, SAFE_QUERY))
    return "&".join(sorted(kept))
