from __future__ import annotations

import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Protocol

from . import ordering

ROUND_SECONDS = 300
BATCH_TARGET = 250
LEASE_TTL_S = 900
MIN_CRAWL_DELAY = 1.0


@dataclass
class Url:
    host: str
    url: str
    crawl_delay: float = MIN_CRAWL_DELAY


@dataclass
class Batch:
    batch_id: str
    urls: list[Url]

    @property
    def hosts(self) -> list[str]:
        return sorted({url.host for url in self.urls})

    def manifest_entry(self) -> dict:
        delays = {url.host: url.crawl_delay for url in self.urls}
        return {
            "batch_id": self.batch_id,
            "hosts": self.hosts,
            "url_count": len(self.urls),
            "crawl_delay": {host: delays[host] for host in self.hosts},
        }


def host_share(crawl_delay: float, batch_target: int = BATCH_TARGET) -> int:
    delay = max(crawl_delay, MIN_CRAWL_DELAY)
    return max(1, min(batch_target, int(ROUND_SECONDS / delay)))


def pack(urls: list[Url], batch_target: int = BATCH_TARGET) -> list[Batch]:
    by_host: dict[str, list[Url]] = defaultdict(list)
    for url in urls:
        by_host[url.host].append(url)

    queues: list[list[Url]] = []
    for host, host_urls in by_host.items():
        share = host_share(host_urls[0].crawl_delay, batch_target)
        for start in range(0, len(host_urls), share):
            queues.append(host_urls[start : start + share])
    queues.sort(key=len, reverse=True)

    batches: list[Batch] = []
    current: list[Url] = []
    for chunk in queues:
        if current and len(current) + len(chunk) > batch_target:
            batches.append(Batch(uuid.uuid4().hex[:16], current))
            current = []
        current.extend(chunk)
    if current:
        batches.append(Batch(uuid.uuid4().hex[:16], current))
    return batches


class SeedSource(Protocol):
    def target_block(self, opened_at: float) -> int: ...

    async def seed_for(self, block: int) -> str | None: ...


@dataclass
class Round:
    round_id: str
    batches: dict[str, Batch]
    manifest_hash: str
    seed_block: int
    opened_at: float
    seed: str | None = None
    order: list[str] = field(default_factory=list)
    closed_at: float | None = None

    @property
    def revealed(self) -> bool:
        return self.seed is not None

    def manifest(self) -> list[dict]:
        return [batch.manifest_entry() for batch in self.batches.values()]

    def public_view(self) -> dict:
        view = {
            "round_id": self.round_id,
            "algorithm": ordering.ALGORITHM,
            "manifest_hash": self.manifest_hash,
            "seed_block": self.seed_block,
            "opened_at": self.opened_at,
            "seed": self.seed,
            "closed_at": self.closed_at,
        }
        if self.closed_at is not None:
            view["manifest"] = sorted(self.manifest(), key=lambda e: e["batch_id"])
            view["serve_order"] = self.order
        return view


def open_round(
    urls: list[Url], seeds: SeedSource, batch_target: int = BATCH_TARGET
) -> Round:
    batches = {batch.batch_id: batch for batch in pack(urls, batch_target)}
    opened_at = time.time()
    return Round(
        round_id=uuid.uuid4().hex[:16],
        batches=batches,
        manifest_hash=ordering.manifest_hash(
            [b.manifest_entry() for b in batches.values()]
        ),
        seed_block=seeds.target_block(opened_at),
        opened_at=opened_at,
    )


def reveal(round_: Round, seed: str) -> list[str]:
    round_.seed = seed
    round_.order = ordering.serve_order(seed, list(round_.batches))
    return round_.order
