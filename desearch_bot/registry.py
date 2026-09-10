"""The shared domain list in Postgres, as one crawler process sees and updates it."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from . import db
from .buckets import Buckets
from .loop import DomainWrite, adopted
from .states import Outcome

SYNCED = "synced"
PAGE = 20_000


@dataclass(frozen=True)
class Change:
    host: str
    rank: int | None
    tld_group: str | None
    state: str
    state_reason: str | None
    categories: list[str] | None


class Registry:
    """Reports visits to Postgres in bulk and brings back what changed in its buckets."""

    def __init__(self, pool, buckets: Buckets):
        self.pool = pool
        self.buckets = buckets
        marks = [store.meta(SYNCED) for store in buckets.stores.values()]
        mark = min((tuple(m) for m in marks if m), default=None)
        self.since = (datetime.fromisoformat(mark[0]), mark[1]) if mark else None

    async def report(
        self, visits: list[tuple[DomainWrite, int]], now: datetime
    ) -> None:
        rows = [
            (w.host, w.state.value, w.reason, w.checked_at, urls, w.canonical_host)
            for w, urls in visits
        ]
        await db.report_visits(self.pool, rows)
        targets = [
            (w.canonical_host, w.host, *adopted(w.canonical_host))
            for w, _ in visits
            if w.visit.outcome is Outcome.REDIRECT and w.canonical_host
        ]
        await db.adopt(self.pool, targets, now)

    async def changes(self) -> list[Change]:
        rows = await db.registry_changes(
            self.pool, self.buckets.stores, self.since, PAGE
        )
        if rows:
            self.since = (rows[-1]["changed_at"], rows[-1]["host"])
            mark = [self.since[0].isoformat(), self.since[1]]
            for store in self.buckets.stores.values():
                store.set_meta(SYNCED, mark)
        return [
            Change(
                r["host"],
                r["rank"],
                r["tld_group"],
                r["state"],
                r["state_reason"],
                r["categories"],
            )
            for r in rows
        ]
