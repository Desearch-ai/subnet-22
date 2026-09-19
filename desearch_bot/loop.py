"""The crawl loop: visit what is due in the buckets this process owns, and keep what it learns."""

from __future__ import annotations

import asyncio
import random
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from . import exclusions, records, states
from .buckets import Buckets, Changes
from .states import Outcome, State
from .suffixes import tld_group
from .timetable import Timetable
from .visit import Known, Visit, Visitor

TICK = 1.0
FLUSH_SECONDS = 5.0
FLUSH_VISITS = 200
# Postgres hears from the loop in bulk, never once per visit.
SYNC_SECONDS = 30.0
CRASH_RETRY = timedelta(hours=1)
# On shutdown, visits still running after this long are dropped; they are simply due again.
STOP_GRACE = 30.0
# A visit cut short by failing requests leaves the site alone this long.
CUT_SHORT_WAIT = timedelta(hours=1)
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
        earliest = now if visit.deferred else _earliest_sitemap(known, visit, now)
        if earliest is not None and (due is None or earliest < due):
            due = earliest
        if visit.cut_short:
            due = max(due, now + CUT_SHORT_WAIT)
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
    """Keeps many visits in flight across the buckets one process owns."""

    def __init__(
        self,
        buckets: Buckets,
        visitor: Visitor,
        concurrency: int,
        registry,
        excluded: frozenset[str],
        report: Callable[[Loop, list[DomainWrite]], None] | None = None,
        clock: Callable[[], datetime] | None = None,
    ):
        self.buckets = buckets
        self.visitor = visitor
        self.concurrency = concurrency
        self.registry = registry
        self.excluded = excluded
        self.report = report
        self.clock = clock or _now
        self.timetable = Timetable()
        self.inflight: dict[str, asyncio.Task] = {}
        self.loaded: dict[str, dict[str, dict]] = {}
        self.pending: list[DomainWrite] = []
        self.unreported: list[tuple[DomainWrite, int]] = []
        self.stopping = asyncio.Event()
        self.visited = 0

    def load(self) -> int:
        """Put every domain the stores hold into the timetable; returns how many are due ever."""
        for store in self.buckets.stores.values():
            for host, record in store.domains():
                self._schedule(host, record)
        return len(self.timetable)

    def stop(self) -> None:
        self.stopping.set()

    async def run(self) -> None:
        await self.sync()
        flushed = synced = time.monotonic()
        while not self.stopping.is_set():
            self._fill()
            try:
                await asyncio.wait_for(self.stopping.wait(), TICK)
            except TimeoutError:
                pass
            now = time.monotonic()
            if len(self.pending) >= FLUSH_VISITS or (
                self.pending and now - flushed >= FLUSH_SECONDS
            ):
                self._flush()
                flushed = now
            if now - synced >= SYNC_SECONDS:
                await self.sync()
                synced = time.monotonic()
        if self.inflight:
            _, late = await asyncio.wait(
                list(self.inflight.values()), timeout=STOP_GRACE
            )
            for task in late:
                task.cancel()
            await asyncio.gather(*late, return_exceptions=True)
        self._flush()
        await self.sync()

    def known(self, host: str) -> Known | None:
        """What a visit needs to know about a domain, from its store; None once excluded."""
        store = self.buckets.store(host)
        record = store.domain(host)
        if record is None or record["state"] == State.EXCLUDED.value:
            return None
        sitemaps = dict(store.sitemaps(host))
        self.loaded[host] = sitemaps
        return records.known(host, record, sitemaps.items())

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

    def save(self, writes: Iterable[DomainWrite]) -> None:
        """Write finished visits to their stores and put each domain back in the timetable."""
        for write in writes:
            sitemaps = self.loaded.pop(write.host, {})
            store = self.buckets.store(write.host)
            current = store.domain(write.host)
            if current is None or current["state"] == State.EXCLUDED.value:
                continue
            changes = Changes()
            for update in write.visit.sitemaps:
                record = records.written_sitemap(
                    sitemaps.get(update.url), update, write.checked_at
                )
                sitemaps[update.url] = record
                changes.sitemap(write.host, update.url, record)
            if write.state is State.ACTIVE:
                for url, depth, parent in write.visit.deferred:
                    if url not in sitemaps:
                        sitemaps[url] = records.unread_sitemap(url, depth, parent)
                        changes.sitemap(write.host, url, sitemaps[url])
            record = records.written_domain(
                current, write, records.url_count(sitemaps.values())
            )
            changes.domain(write.host, record)
            store.write(changes)
            self._schedule(write.host, record)
            self.unreported.append((write, record["urls"]))

    async def sync(self) -> None:
        """Send the registry what visits found, and take in what changed there."""
        visits, self.unreported = self.unreported, []
        await self.registry.report(visits, self.clock())
        for change in await self.registry.changes():
            self._apply(change)

    def _apply(self, change) -> None:
        store = self.buckets.store(change.host)
        record = store.domain(change.host)
        excluded = change.state == State.EXCLUDED.value
        if record is None:
            record = records.new_domain(
                change.rank,
                change.tld_group,
                change.categories or (),
                state=State.EXCLUDED if excluded else State.NEW,
                reason=change.state_reason if excluded else None,
                due=None if excluded else self.clock(),
            )
        else:
            record = dict(
                record,
                rank=change.rank,
                group=change.tld_group,
                categories=sorted(change.categories or ()),
            )
            if excluded:
                record.update(
                    state=State.EXCLUDED.value, reason=change.state_reason, due=None
                )
        changes = Changes()
        changes.domain(change.host, record)
        store.write(changes)
        if change.host not in self.inflight:
            self._schedule(change.host, record)

    def _schedule(self, host: str, record: dict) -> None:
        self.timetable.set(
            host,
            State(record["state"]),
            records.moment(record.get("due")),
            record.get("rank"),
        )

    def _fill(self) -> None:
        free = self.concurrency - len(self.inflight)
        if free <= 0:
            return
        for host in self.timetable.take(free, self.clock()):
            known = self.known(host)
            if known is not None:
                self.inflight[host] = asyncio.create_task(self._one(known))

    async def _one(self, known: Known) -> None:
        # Await before touching the list: a flush during the visit swaps in a new one.
        write = await self.once(known)
        self.pending.append(write)
        self.inflight.pop(known.host, None)
        self.visited += 1

    def _flush(self) -> None:
        if not self.pending:
            return
        writes, self.pending = self.pending, []
        self.save(writes)
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
