from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field

from . import proofs

BATCH_TARGET = 250
LEASE_TTL_S = 900


@dataclass
class Url:
    host: str
    url: str


@dataclass
class Batch:
    batch_id: str
    urls: list[Url]

    def urls_hash(self) -> str:
        return proofs.sha256(proofs.canonical_json([u.url for u in self.urls]))

    def manifest_entry(self) -> dict:
        return {
            "batch_id": self.batch_id,
            "url_count": len(self.urls),
            "urls_hash": self.urls_hash(),
        }


def spread(urls: list[Url]) -> list[Url]:
    """Round-robin hosts so a refusing site costs a row, not a task."""
    by_host: dict[str, list[Url]] = {}
    for url in urls:
        by_host.setdefault(url.host, []).append(url)
    queues = list(by_host.values())
    mixed = []
    while queues:
        queues = [queue for queue in queues if queue]
        for queue in queues:
            mixed.append(queue.pop(0))
    return mixed


def pack(urls: list[Url], batch_target: int = BATCH_TARGET) -> list[Batch]:
    mixed = spread(urls)
    return [
        Batch(uuid.uuid4().hex[:16], mixed[i : i + batch_target])
        for i in range(0, len(mixed), batch_target)
    ]


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
            "algorithm": proofs.ALGORITHM,
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
    urls: list[Url], seed_block: int, batch_target: int = BATCH_TARGET
) -> Round:
    batches = {batch.batch_id: batch for batch in pack(urls, batch_target)}
    opened_at = time.time()
    return Round(
        round_id=uuid.uuid4().hex[:16],
        batches=batches,
        manifest_hash=proofs.manifest_hash(
            [b.manifest_entry() for b in batches.values()], seed_block
        ),
        seed_block=seed_block,
        opened_at=opened_at,
    )


def reveal(round_: Round, seed: str) -> list[str]:
    round_.seed = seed
    round_.order = proofs.serve_order(seed, list(round_.batches))
    return round_.order
