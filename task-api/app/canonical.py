from __future__ import annotations

import hashlib
import html
import re
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

TRACKING = re.compile(r"^(utm_|fbclid$|gclid$|mc_cid$|mc_eid$|ref$|cmpid$|ito$)", re.I)


def canonicalize(url: str) -> str:
    u = html.unescape((url or "").strip())
    parts = urlsplit(u)
    query = [
        (k, v)
        for k, v in parse_qsl(parts.query, keep_blank_values=True)
        if not TRACKING.match(k)
    ]
    return urlunsplit(
        (parts.scheme, parts.netloc.lower(), parts.path, urlencode(query), "")
    )


def domain_of(url: str) -> str:
    host = urlsplit(url).netloc.lower()
    return host[4:] if host.startswith("www.") else host


def url_sha1(url: str) -> str:
    return hashlib.sha1(canonicalize(url).encode()).hexdigest()


def article_key(url: str) -> str:
    return f"articles/{domain_of(url)}/{url_sha1(url)}"
