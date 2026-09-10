"""Domains split into fixed buckets, each bucket kept in its own RocksDB store."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Iterator
from pathlib import Path

from rocksdict import (
    BlockBasedOptions,
    Cache,
    DBCompressionType,
    Options,
    Rdict,
    ReadOptions,
    WriteBatch,
    WriteBufferManager,
)

from .urls import HTTPS, TIMED, WWW, Listing, Record, Url

BUCKETS = 256
DOMAIN = b"D"
SITEMAP = b"S"
URL = b"U"
META = b"M"


def bucket_of(host: str) -> int:
    """The bucket a domain belongs to; it never changes."""
    digest = hashlib.blake2b(host.encode(), digest_size=2).digest()
    return int.from_bytes(digest, "big") % BUCKETS


def sitemap_id(url: str) -> int:
    """A stable id for a sitemap file, the same wherever it is computed."""
    digest = hashlib.blake2b(url.encode(), digest_size=8).digest()
    return int.from_bytes(digest, "big") >> 1


def owned_buckets(worker: int, workers: int) -> list[int]:
    """The buckets one of this many crawler processes owns."""
    return [bucket for bucket in range(BUCKETS) if bucket % workers == worker]


class Resources:
    """The block cache and memtable budget shared by every store one process opens."""

    def __init__(self, cache_bytes: int = 512 << 20, memtable_bytes: int = 512 << 20):
        self.cache = Cache(cache_bytes)
        self.memtables = WriteBufferManager(memtable_bytes, False)

    def options(self) -> Options:
        table = BlockBasedOptions()
        table.set_block_cache(self.cache)
        table.set_bloom_filter(10, False)
        options = Options(raw_mode=True)
        options.create_if_missing(True)
        options.set_block_based_table_factory(table)
        options.set_compression_type(DBCompressionType.zstd())
        options.set_write_buffer_size(16 << 20)
        options.set_max_write_buffer_number(3)
        options.set_write_buffer_manager(self.memtables)
        options.set_level_compaction_dynamic_level_bytes(True)
        options.set_max_background_jobs(2)
        options.set_max_open_files(512)
        return options


class Changes:
    """Domain and sitemap records bound for one store, written together."""

    def __init__(self):
        self.batch = WriteBatch(raw_mode=True)

    def domain(self, host: str, record: dict) -> None:
        self.batch.put(DOMAIN + host.encode(), _encode(record))

    def sitemap(self, host: str, url: str, record: dict) -> None:
        self.batch.put(_sitemap_key(host, url), _encode(record))


class BucketStore:
    """One bucket's domains, their sitemaps and every URL they list."""

    def __init__(self, path: Path, resources: Resources):
        self.path = Path(path)
        self.db = Rdict(str(self.path), resources.options())

    def __enter__(self) -> BucketStore:
        return self

    def __exit__(self, *_) -> None:
        self.close()

    def close(self) -> None:
        self.db.close()

    def write(self, changes: Changes) -> None:
        self.db.write(changes.batch)

    def meta(self, name: str):
        raw = self.db.get(META + name.encode())
        return None if raw is None else json.loads(raw)

    def set_meta(self, name: str, value) -> None:
        self.db.put(META + name.encode(), _encode(value))

    def domain(self, host: str) -> dict | None:
        raw = self.db.get(DOMAIN + host.encode())
        return None if raw is None else json.loads(raw)

    def domains(self) -> Iterator[tuple[str, dict]]:
        for key, raw in self._scan(DOMAIN):
            yield key[1:].decode(), json.loads(raw)

    def sitemaps(self, host: str) -> Iterator[tuple[str, dict]]:
        prefix = _sitemap_key(host, "")
        for key, raw in self._scan(prefix):
            yield key[len(prefix) :].decode(), json.loads(raw)

    def record_listing(
        self, sitemap_id: int, entries: list[tuple[Url, int, bool]], now: int
    ) -> Listing:
        """Store what one sitemap lists right now; each URL it names becomes its own."""
        unique: dict[bytes, tuple[Url, int, bool]] = {}
        for url, lastmod, timed in entries:
            unique.setdefault(URL + url.key, (url, lastmod, timed))
        keys = list(unique)
        if not keys:
            return Listing(0, 0, 0)

        batch = WriteBatch(raw_mode=True)
        new = moved = 0
        for key, raw in zip(keys, self.db.get(keys)):
            url, lastmod, timed = unique[key]
            if raw is None:
                flags = url.flags | (TIMED if timed else 0)
                record = Record(sitemap_id, lastmod, now, now, flags=flags)
                new += 1
            else:
                record = Record.unpack(raw)
                if lastmod and lastmod != record.lastmod:
                    record.lastmod = lastmod
                    record.flags = (record.flags & ~TIMED) | (TIMED if timed else 0)
                    moved += 1
                record.sitemap_id = sitemap_id
                record.last_seen = now
            batch.put(key, record.pack())
        self.db.write(batch)
        return Listing(len(keys), new, moved)

    def url(self, url: Url) -> Record | None:
        raw = self.db.get(URL + url.key)
        return None if raw is None else Record.unpack(raw)

    def urls(self, domain: str) -> Iterator[tuple[Url, Record]]:
        """Every URL stored for a domain, in key order."""
        for key, raw in self._scan(URL + domain.encode() + b"\x00"):
            record = Record.unpack(raw)
            yield Url(key[1:], record.flags & (HTTPS | WWW)), record

    def estimate(self) -> int:
        return self.db.property_int_value("rocksdb.estimate-num-keys") or 0

    def _scan(self, prefix: bytes) -> Iterator[tuple[bytes, bytes]]:
        bounds = ReadOptions()
        bounds.set_iterate_upper_bound(prefix[:-1] + bytes([prefix[-1] + 1]))
        for key, raw in self.db.items(from_key=prefix, read_opt=bounds):
            if not key.startswith(prefix):
                return
            yield key, raw


class Buckets:
    """The stores of the buckets one process owns."""

    def __init__(self, root: Path, owned: Iterable[int], resources: Resources):
        root = Path(root)
        self.stores = {b: BucketStore(root / f"{b:03d}", resources) for b in owned}

    def __enter__(self) -> Buckets:
        return self

    def __exit__(self, *_) -> None:
        self.close()

    def owns(self, host: str) -> bool:
        return bucket_of(host) in self.stores

    def store(self, host: str) -> BucketStore:
        return self.stores[bucket_of(host)]

    def close(self) -> None:
        for store in self.stores.values():
            store.close()


def _sitemap_key(host: str, url: str) -> bytes:
    return SITEMAP + host.encode() + b"\x00" + url.encode()


def _encode(value) -> bytes:
    return json.dumps(value, separators=(",", ":")).encode()
