"""Page URLs canonicalised and keyed as the task API does (`crates/desearch/src/canonical.rs`), for the URLs the feeder has sent."""

from __future__ import annotations

import hashlib
import re
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

TRACKING = re.compile(r"^(utm_|fbclid$|gclid$|mc_cid$|mc_eid$|ref$|cmpid$|ito$)", re.I)


def canonicalize(url: str) -> str:
    parts = urlsplit((url or "").strip())
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
