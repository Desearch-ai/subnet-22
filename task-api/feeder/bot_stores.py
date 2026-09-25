from __future__ import annotations

import hashlib
import heapq
import struct
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

BUCKETS = 256
BUCKETS_DIR = "/mnt/desearch-bot/buckets"
SECONDARY_DIR = "/var/lib/desearch-bot/tmp/task-api-feeder"
URL = b"U"
RECORD = struct.Struct("<qIIIIIB")
HTTPS = 1
WWW = 2


@dataclass(frozen=True)
class ListedPage:
    url: str
    lastmod: int
    first_seen: int
    crawled_at: int


def store_index(host: str) -> int:
    digest = hashlib.blake2b(host.encode(), digest_size=2).digest()
    return int.from_bytes(digest, "big") % BUCKETS


def host_prefix(host: str) -> bytes:
    return URL + host.encode() + b"\x00"


def range_end(prefix: bytes) -> bytes:
    return prefix[:-1] + bytes([prefix[-1] + 1])


def parse_listed_page(key: bytes, raw: bytes) -> ListedPage:
    _, lastmod, first_seen, _, crawled_at, _, flags = RECORD.unpack(raw)
    rest = key[1:].split(b"\x00", 1)[1].decode()
    return ListedPage(
        f"{'https' if flags & HTTPS else 'http'}://{'www.' if flags & WWW else ''}{rest}",
        lastmod,
        first_seen,
        crawled_at,
    )


def open_store(root: str, bucket: int, secondary: str):
    from rocksdict import AccessType, Options, Rdict

    name = f"{bucket:03d}"
    Path(secondary).mkdir(parents=True, exist_ok=True)
    store = Rdict(
        str(Path(root) / name),
        Options(raw_mode=True),
        access_type=AccessType.secondary(str(Path(secondary) / name)),
    )
    store.try_catch_up_with_primary()
    return store


def newest_pages(store, host: str, keep: int) -> list[ListedPage]:
    """Capped per domain, even for sites that stamp every page today."""
    from rocksdict import ReadOptions

    prefix = host_prefix(host)
    bounds = ReadOptions()
    bounds.set_iterate_upper_bound(range_end(prefix))
    best: list[tuple] = []
    for key, raw in store.items(from_key=prefix, read_opt=bounds):
        if not key.startswith(prefix):
            break
        _, lastmod, first_seen, *_ = RECORD.unpack(raw)
        entry = (lastmod, first_seen, bytes(key), bytes(raw))
        if len(best) < keep:
            heapq.heappush(best, entry)
        elif entry[:2] > best[0][:2]:
            heapq.heapreplace(best, entry)
    return [
        parse_listed_page(key, raw) for _, _, key, raw in sorted(best, reverse=True)
    ]


def collect_pages(
    root: str, secondary: str, hosts: list[str], per_domain: int
) -> list[dict]:
    by_bucket: dict[int, list[str]] = defaultdict(list)
    for host in hosts:
        by_bucket[store_index(host)].append(host)

    rows = []
    for bucket, owned in sorted(by_bucket.items()):
        store = open_store(root, bucket, secondary)
        try:
            for host in owned:
                rows += [
                    {"host": host, "url": page.url, "lastmod": page.lastmod}
                    for page in newest_pages(store, host, per_domain)
                ]
        finally:
            store.close()
    return rows
