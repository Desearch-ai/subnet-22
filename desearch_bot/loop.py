"""The crawl loop: visit whatever is due, forever, and write down what each visit learned."""

from __future__ import annotations

import asyncio
import random
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from . import db, exclusions, states
from .states import Outcome, State
from .suffixes import tld_group
from .visit import Known, Visit, Visitor

TICK = 2.0
FLUSH_SECONDS = 5.0
FLUSH_VISITS = 200
CRASH_RETRY = timedelta(hours=1)
ANSWERED = frozenset(
    {
        Outcome.SITEMAP,
        Outcome.NO_SITEMAP,
        Outcome.REDIRECT,
        Outcome.BLOCKED,
        Outcome.INELIGIBLE,
    }
)


@dataclass(frozen=True)
class DomainWrite:
    host: str
    state: State
    reason: str | None
    failures: int
    next_due_at: datetime | None
    last_ok_at: datetime | None
    canonical_host: str | None
    checked_at: datetime
    visit: Visit


def plan(
    known: Known, visit: Visit, now: datetime, rng: random.Random | None = None
) -> DomainWrite:
    """The state a visit leaves a domain in, and when the loop should come back to it."""
    decision = states.decide(
        known.state, visit.outcome, known.failures, known.last_ok_at, now, rng
    )
    due = decision.next_check_at
    if decision.state is State.ACTIVE:
        earliest = _earliest_sitemap(known, visit, now)
        if earliest is not None and (due is None or earliest < due):
            due = earliest
    if visit.outcome is Outcome.REDIRECT:
        canonical = visit.canonical_host
    elif visit.outcome in ANSWERED:
        canonical = None
    else:
        canonical = known.canonical_host
    return DomainWrite(
        known.host,
        decision.state,
        visit.reason,
        decision.failures,
        due,
        now if visit.outcome in ANSWERED and visit.requests else known.last_ok_at,
        canonical,
        now,
        visit,
    )


def crashed(known: Known, error: str, now: datetime) -> DomainWrite:
    """A visit that failed on our side leaves the domain as it was, to be tried again later."""
    return DomainWrite(
        known.host,
        known.state,
        f"crashed: {error}",
        known.failures,
        now + CRASH_RETRY,
        known.last_ok_at,
        known.canonical_host,
        now,
        Visit(known.host),
    )


def adopted(target: str) -> tuple[str, State, str | None]:
    """How a domain reached through a redirect joins the list: its group, state, and why."""
    group = tld_group(target)
    reason = exclusions.exclusion_reason(target, {}, group)
    return group, State.EXCLUDED if reason else State.NEW, reason


class Loop:
    """Keeps a fixed number of visits in flight, refreshing known domains before discovering."""

    def __init__(
        self,
        pool,
        visitor: Visitor,
        concurrency: int,
        excluded: frozenset[str],
        report: Callable[[Loop, list[DomainWrite]], None] | None = None,
        clock: Callable[[], datetime] | None = None,
    ):
        self.pool = pool
        self.visitor = visitor
        self.concurrency = concurrency
        self.excluded = excluded
        self.report = report
        self.clock = clock or _now
        self.inflight: dict[str, asyncio.Task] = {}
        self.pending: list[DomainWrite] = []
        self.stopping = asyncio.Event()
        self.visited = 0

    def stop(self) -> None:
        self.stopping.set()

    async def run(self) -> None:
        flushed = time.monotonic()
        while not self.stopping.is_set():
            await self._fill()
            try:
                await asyncio.wait_for(self.stopping.wait(), TICK)
            except TimeoutError:
                pass
            stale = time.monotonic() - flushed >= FLUSH_SECONDS
            if len(self.pending) >= FLUSH_VISITS or (self.pending and stale):
                await self._flush()
                flushed = time.monotonic()
        if self.inflight:
            await asyncio.gather(*self.inflight.values(), return_exceptions=True)
        await self._flush()

    async def _fill(self) -> None:
        free = self.concurrency - len(self.inflight)
        if free <= 0:
            return
        now, busy = self.clock(), list(self.inflight)
        batch = await db.due(self.pool, free, busy, True, now)
        busy += [known.host for known in batch]
        batch += await db.due(self.pool, free - len(batch), busy, False, now)
        for known in batch:
            self.inflight[known.host] = asyncio.create_task(self._one(known))

    async def once(self, known: Known) -> DomainWrite:
        """Visit one domain and decide what comes next; a failure on our side never escapes."""
        try:
            if known.categories & self.excluded or exclusions.blocked_operator(
                known.host
            ):
                visit = Visit(known.host, Outcome.EXCLUDED, reason="excluded")
            else:
                visit = await self.visitor.visit(known, self.clock())
            return plan(known, visit, self.clock())
        except Exception as exc:
            return crashed(known, type(exc).__name__, self.clock())

    async def save(self, writes: list[DomainWrite]) -> None:
        """Record finished visits, and add the domains they redirect to."""
        await db.save_visits(self.pool, writes)
        targets = [
            (w.canonical_host, w.host, *adopted(w.canonical_host))
            for w in writes
            if w.visit.outcome is Outcome.REDIRECT and w.canonical_host
        ]
        await db.adopt(self.pool, targets, self.clock())

    async def _one(self, known: Known) -> None:
        self.pending.append(await self.once(known))
        self.inflight.pop(known.host, None)
        self.visited += 1

    async def _flush(self) -> None:
        if not self.pending:
            return
        writes, self.pending = self.pending, []
        await self.save(writes)
        if self.report:
            self.report(self, writes)


def _earliest_sitemap(known: Known, visit: Visit, now: datetime) -> datetime | None:
    updated = {update.url: update.next_check_at for update in visit.sitemaps}
    times = [
        updated.get(s.url, s.next_check_at) or now for s in known.sitemaps.values()
    ]
    times += [
        when for url, when in updated.items() if url not in known.sitemaps and when
    ]
    return min(times) if times else None


def _now() -> datetime:
    return datetime.now(timezone.utc)
