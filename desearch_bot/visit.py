"""One visit to one domain: read its rules, read its sitemaps, keep what is new."""

from __future__ import annotations

import asyncio
import hashlib
import math
import time
import zlib
from collections import Counter, deque
from collections.abc import Awaitable, Callable, Iterable, Mapping
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from urllib.parse import urljoin, urlsplit

import aiohttp

from . import homepage, net, robots, schedule, signing, sitemaps
from .schedule import Trust
from .states import JITTER, Outcome, State
from .urls import UrlStore, parse

MIN_HOST_INTERVAL = 1.0
# Some sites ask for hours between requests; past this we slow down no further.
MAX_CRAWL_DELAY = 60.0
MAX_REDIRECTS = 5
REDIRECT_STATUSES = frozenset({301, 302, 303, 307, 308})
GONE_STATUSES = frozenset({404, 410})
SLOW_DOWN_STATUSES = frozenset({429, 503})
SITEMAP_GUESSES = ("/sitemap.xml", "/sitemap_index.xml")
MAX_DEPTH = 3
MAX_FILES = 40
# The most unread sitemap files one visit records for the visits after it.
MAX_DEFERRED = 10_000
MIN_URLS = 10
ROBOTS_EVERY = timedelta(days=1)
MAX_ROBOTS_BYTES = 512 * 1024
MAX_HOMEPAGE_BYTES = 512 * 1024
# The sitemap protocol caps a file at 50 MB uncompressed, and so do we.
MAX_SITEMAP_BYTES = 50 * 1024 * 1024

Allocate = Callable[[str, str], Awaitable[int]]


@dataclass
class KnownSitemap:
    id: int
    url: str
    kind: str | None
    depth: int
    parent_id: int | None
    etag: str | None
    last_modified: str | None
    content_hash: str | None
    interval: timedelta
    next_check_at: datetime | None
    trust: Trust
    index_lastmod: str | None
    url_count: int = 0


@dataclass
class Known:
    host: str
    state: State = State.NEW
    failures: int = 0
    last_ok_at: datetime | None = None
    robots_checked_at: datetime | None = None
    robots_allows: bool | None = None
    crawl_delay: float | None = None
    language: str | None = None
    categories: frozenset[str] = frozenset()
    sitemaps: dict[str, KnownSitemap] = field(default_factory=dict)
    canonical_host: str | None = None


@dataclass
class SitemapUpdate:
    id: int
    url: str
    kind: str | None
    depth: int
    parent_id: int | None
    status: str = "ok"
    error: str | None = None
    etag: str | None = None
    last_modified: str | None = None
    content_hash: str | None = None
    changed: bool = False
    url_count: int = 0
    child_count: int = 0
    interval: timedelta = schedule.DEFAULT_INTERVAL
    next_check_at: datetime | None = None
    trust: Trust = Trust.UNKNOWN
    index_lastmod: str | None = None


@dataclass
class Visit:
    host: str
    outcome: Outcome = Outcome.SITEMAP
    reason: str | None = None
    canonical_host: str | None = None
    robots_read: bool = False
    robots_status: int | None = None
    robots_allows: bool | None = None
    crawl_delay: float | None = None
    language: str | None = None
    declared_lang: str | None = None
    home_chars: int | None = None
    sitemaps: list[SitemapUpdate] = field(default_factory=list)
    listed: int = 0
    new: int = 0
    moved: int = 0
    requests: int = 0
    deferred: list[tuple[str, int, int | None]] = field(default_factory=list)


@dataclass(frozen=True)
class Answer:
    status: int
    body: bytes
    headers: dict[str, str]
    url: str


class TooManyRedirects(Exception):
    """A redirect chain longer than we are willing to follow."""


class Pacer:
    """One host's request clock, so a slow host never shares a budget with a fast one."""

    def __init__(self, delay: float | None = None, floor: float = MIN_HOST_INTERVAL):
        self.interval = max(_delay(delay), floor)
        self._next = 0.0

    def slow_to(self, delay: float | None) -> None:
        """Adopt a longer interval, pushing back a request already booked at the shorter one."""
        delay = _delay(delay)
        if delay <= self.interval:
            return
        if self._next:
            self._next += delay - self.interval
        self.interval = delay

    def rest(self, seconds: float) -> None:
        """Hold the next request back at least this long from now."""
        self._next = max(self._next, time.monotonic() + min(seconds, MAX_CRAWL_DELAY))

    async def wait(self) -> None:
        remaining = self._next - time.monotonic()
        if remaining > 0:
            await asyncio.sleep(remaining)
        self._next = time.monotonic() + self.interval


class Visitor:
    """Everything a visit needs besides the domain itself."""

    def __init__(
        self,
        session: aiohttp.ClientSession,
        store: UrlStore,
        allocate: Allocate,
        detect_language: Callable[[str], str | None],
        registrable: Callable[[str], str | None],
        signer: signing.Signer | None = None,
        timeout: float = 10.0,
        floor: float = MIN_HOST_INTERVAL,
    ):
        self.session = session
        self.store = store
        self.allocate = allocate
        self.detect_language = detect_language
        self.registrable = registrable
        self.signer = signer
        self.floor = floor
        self.timeout = aiohttp.ClientTimeout(
            total=timeout, connect=min(timeout, 6.0), sock_connect=min(timeout, 6.0)
        )

    async def visit(self, known: Known, now: datetime) -> Visit:
        """Visit one domain and report everything the visit learned."""
        run = _Run(self, known, now)
        await run.go()
        return run.result


class _Run:
    def __init__(self, visitor: Visitor, known: Known, now: datetime):
        self.visitor = visitor
        self.known = known
        self.now = now
        self.pacer = Pacer(known.crawl_delay, visitor.floor)
        self.result = Visit(
            known.host, robots_allows=known.robots_allows, crawl_delay=known.crawl_delay
        )
        self.guesses: set[str] = set()
        self.found = False
        self.fetches = 0
        self.kept = 0
        self.answered = False
        self.network_error: str | None = None

    async def go(self) -> None:
        roots = await self._robots() if self._robots_due() else self._roots([])
        if roots is None:
            return
        await self._walk(roots)
        self._settle()
        if self.result.outcome is Outcome.SITEMAP and self._needs_language():
            await self._language()

    def _robots_due(self) -> bool:
        checked = self.known.robots_checked_at
        return (
            self.known.state is not State.ACTIVE
            or checked is None
            # The daily recheck can land up to 10% early and should still read robots.txt.
            or self.now - checked >= ROBOTS_EVERY * (1 - JITTER)
        )

    def _roots(self, named: list[str]) -> list[str]:
        base = f"https://{self.known.host}/"
        known = [s.url for s in self.known.sitemaps.values() if s.depth == 0]
        roots = list(dict.fromkeys([urljoin(base, url) for url in named] + known))
        if roots:
            return roots
        self.guesses = {urljoin(base, path) for path in SITEMAP_GUESSES}
        return [urljoin(base, path) for path in SITEMAP_GUESSES]

    async def _robots(self) -> list[str] | None:
        """Read robots.txt; None means the visit ends here."""
        answer = await self._first_answer("/robots.txt", MAX_ROBOTS_BYTES)
        if answer is None or self._moved_away(answer):
            return None
        self.result.robots_read = True
        self.result.robots_status = answer.status
        if answer.status >= 500:
            return self._stop(Outcome.UNREACHABLE, f"robots_{answer.status}")

        allowed, delay, named = True, None, []
        if answer.status == 200:
            text = homepage.decode(answer.body)
            allowed, delay = robots.rules(text)
            named = robots.sitemaps(text)
        self.result.robots_allows, self.result.crawl_delay = allowed, delay
        self.pacer.slow_to(delay)
        if not allowed:
            return self._stop(Outcome.BLOCKED, "robots_disallow")
        return self._roots(named)

    async def _walk(self, roots: list[str]) -> None:
        queue: deque[tuple[str, int, int | None, str | None, bool]] = deque(
            (url, 0, None, None, False) for url in roots
        )
        queue.extend(
            (s.url, s.depth, s.parent_id, None, False)
            for s in self.known.sitemaps.values()
            if s.depth > 0 and _due(s, self.now)
        )
        seen: set[str] = set()
        while queue and self.fetches < MAX_FILES:
            url, depth, parent_id, index_date, force = queue.popleft()
            if url in seen or (url in self.guesses and self.found):
                continue
            seen.add(url)
            stored = self.known.sitemaps.get(url)
            if stored and not force and not _due(stored, self.now):
                continue
            queue.extend(await self._sitemap(url, depth, parent_id, index_date, stored))
        for url, depth, parent_id, _, _ in queue:
            if len(self.result.deferred) >= MAX_DEFERRED:
                break
            if (
                url not in seen
                and url not in self.known.sitemaps
                and url not in self.guesses
            ):
                seen.add(url)
                self.result.deferred.append((url, depth, parent_id))

    async def _sitemap(
        self,
        url: str,
        depth: int,
        parent_id: int | None,
        index_date: str | None,
        stored: KnownSitemap | None,
    ) -> list[tuple[str, int, int | None, str | None, bool]]:
        """Read one sitemap file and return the child sitemaps worth reading next."""
        self.fetches += 1
        validators = (stored.etag, stored.last_modified) if stored else None
        try:
            answer, error = await self._get(url, MAX_SITEMAP_BYTES, validators), None
        except Exception as exc:
            answer, error = None, type(exc).__name__
            self.network_error = error
        if url in self.guesses and (answer is None or answer.status != 200):
            return []

        update = await self._update(url, depth, parent_id, index_date, stored)
        if answer is None:
            return self._failed(update, error)
        if answer.status == 304:
            return self._unchanged(update, stored)
        if answer.status in GONE_STATUSES:
            update.status = "gone"
            update.next_check_at = self.now + schedule.MAX_INTERVAL
            return []
        if answer.status != 200 or not answer.body:
            return self._failed(update, f"http_{answer.status}")

        body = _gunzip(answer.body)
        digest = hashlib.sha256(body).hexdigest()[:32]
        update.etag = answer.headers.get("ETag")
        update.last_modified = answer.headers.get("Last-Modified")
        if stored and digest == stored.content_hash:
            return self._unchanged(update, stored)
        kind, entries = sitemaps.parse_entries(body)
        if kind == "invalid":
            return self._failed(update, "invalid")

        self.found = True
        update.kind, update.content_hash, update.changed = kind, digest, True
        dates = [
            schedule.plausible(
                sitemaps.parse_lastmod(e.lastmod or e.published), self.now
            )
            for e in entries
        ]
        update.trust = schedule.assess_dates(dates, self.now, update.trust)
        if stored:
            update.interval = schedule.next_interval(update.interval, changed=True)
        else:
            news = (
                sitemaps.is_news(body)
                or "news" in url.lower()
                or "news" in self.known.categories
            )
            update.interval = schedule.first_interval(
                _dominant(e.changefreq for e in entries), news
            )
        update.next_check_at = self.now + update.interval

        if kind == "index":
            update.child_count = len(entries)
            return self._children(update, entries, depth)
        await self._record(update, entries, dates)
        return []

    async def _update(
        self,
        url: str,
        depth: int,
        parent_id: int | None,
        index_date: str | None,
        stored: KnownSitemap | None,
    ) -> SitemapUpdate:
        if stored:
            update = SitemapUpdate(
                stored.id,
                url,
                stored.kind,
                depth,
                parent_id,
                etag=stored.etag,
                last_modified=stored.last_modified,
                content_hash=stored.content_hash,
                interval=stored.interval,
                trust=stored.trust,
                index_lastmod=index_date or stored.index_lastmod,
            )
        else:
            sitemap_id = await self.visitor.allocate(self.known.host, url)
            update = SitemapUpdate(
                sitemap_id, url, None, depth, parent_id, index_lastmod=index_date
            )
        self.result.sitemaps.append(update)
        return update

    def _children(
        self, update: SitemapUpdate, entries: list[sitemaps.Entry], depth: int
    ) -> list[tuple[str, int, int | None, str | None, bool]]:
        if depth >= MAX_DEPTH:
            return []
        trusted = schedule.relies_on_dates(update.trust)
        children = []
        for entry in entries:
            url = urljoin(update.url, entry.url)
            stored = self.known.sitemaps.get(url)
            if (
                stored
                and trusted
                and entry.lastmod
                and entry.lastmod == stored.index_lastmod
            ):
                continue
            force = stored is None or (
                trusted and entry.lastmod != stored.index_lastmod
            )
            children.append((url, depth + 1, update.id, entry.lastmod, force))
        return children

    async def _record(
        self,
        update: SitemapUpdate,
        entries: list[sitemaps.Entry],
        dates: list[datetime | None],
    ) -> None:
        rows = []
        for entry, date in zip(entries, dates):
            url = parse(entry.url, self.known.host)
            if url is not None:
                rows.append(
                    (
                        url,
                        _epoch(date),
                        sitemaps.has_time(entry.lastmod or entry.published),
                    )
                )
        listing = await asyncio.to_thread(
            self.visitor.store.record_listing, update.id, rows, _epoch(self.now)
        )
        update.url_count = listing.listed
        self.result.listed += listing.listed
        self.result.new += listing.new
        self.result.moved += listing.moved

    def _unchanged(self, update: SitemapUpdate, stored: KnownSitemap | None) -> list:
        self.found = True
        self.kept += stored.url_count if stored else 0
        update.interval = schedule.next_interval(update.interval, changed=False)
        update.next_check_at = self.now + update.interval
        return []

    def _failed(self, update: SitemapUpdate, error: str | None) -> list:
        update.status, update.error = "error", error
        update.next_check_at = self.now + update.interval
        return []

    def _settle(self) -> None:
        if self.result.outcome is not Outcome.SITEMAP:
            return
        updates = self.result.sitemaps
        if self.known.state is State.ACTIVE:
            roots = [u for u in updates if u.depth == 0]
            if self.result.requests and not self.answered:
                self._stop(Outcome.UNREACHABLE, self.network_error)
            elif roots and not self.found and all(u.status == "gone" for u in roots):
                self._stop(Outcome.NO_SITEMAP, "sitemap_gone")
            return
        if not self.found:
            self._stop(Outcome.NO_SITEMAP, "no_sitemap")
        elif self.result.listed + self.kept < MIN_URLS and not any(
            u.kind == "index" for u in updates
        ):
            self._stop(Outcome.NO_SITEMAP, "sitemap_too_small")

    def _needs_language(self) -> bool:
        return self.known.language is None or self.known.state is State.INELIGIBLE

    async def _language(self) -> None:
        answer = await self._first_answer("/", MAX_HOMEPAGE_BYTES)
        if answer is None or self._moved_away(answer):
            return
        if answer.status >= 500:
            self._stop(Outcome.UNREACHABLE, f"homepage_{answer.status}")
            return
        if answer.status != 200:
            self._stop(Outcome.INELIGIBLE, "homepage_error")
            return
        page = homepage.read(answer.body, self.visitor.detect_language)
        self.result.home_chars = page.chars
        self.result.declared_lang = page.declared
        self.result.language = page.language
        if page.problem:
            self._stop(Outcome.INELIGIBLE, page.problem)

    async def _first_answer(self, path: str, limit: int) -> Answer | None:
        """Try https, then http; stop at a DNS failure since the other scheme will not help."""
        last = "unknown"
        for scheme in ("https", "http"):
            try:
                return await self._get(f"{scheme}://{self.known.host}{path}", limit)
            except Exception as exc:
                last = type(exc).__name__
                if "DNS" in last:
                    break
        return self._stop(Outcome.UNREACHABLE, last)

    def _moved_away(self, answer: Answer) -> bool:
        target = self.visitor.registrable(urlsplit(answer.url).hostname or "")
        if not target or target == self.known.host:
            return False
        self.result.canonical_host = target
        self._stop(Outcome.REDIRECT, "redirect")
        return True

    async def _get(
        self,
        url: str,
        limit: int,
        validators: tuple[str | None, str | None] | None = None,
    ) -> Answer:
        """Follow redirects by hand, pacing and signing each hop for the host it goes to."""
        conditional = {}
        if validators:
            etag, modified = validators
            if etag:
                conditional["If-None-Match"] = etag
            if modified:
                conditional["If-Modified-Since"] = modified
        for _ in range(MAX_REDIRECTS + 1):
            if not net.public_host(urlsplit(url).hostname or ""):
                raise OSError("refusing a non-public address")
            await self.pacer.wait()
            self.result.requests += 1
            headers = {
                **signing.request_headers(url, self.visitor.signer),
                **conditional,
            }
            started = time.monotonic()
            async with self.visitor.session.get(
                url,
                timeout=self.visitor.timeout,
                allow_redirects=False,
                headers=headers,
            ) as response:
                self.answered = True
                waited = time.monotonic() - started
                if response.status in SLOW_DOWN_STATUSES:
                    self.pacer.slow_to(
                        max(self.pacer.interval * 2, _retry_after(response.headers))
                    )
                location = response.headers.get("Location")
                if response.status in REDIRECT_STATUSES and location:
                    url = urljoin(url, location)
                    continue
                body = (
                    await response.content.read(limit)
                    if response.status == 200
                    else b""
                )
                # Slow to answer means busy: rest that long before the next request.
                self.pacer.rest(waited)
                return Answer(response.status, body, dict(response.headers), url)
        raise TooManyRedirects(url)

    def _stop(self, outcome: Outcome, reason: str | None) -> None:
        self.result.outcome, self.result.reason = outcome, reason
        return None


def _due(stored: KnownSitemap, now: datetime) -> bool:
    return stored.next_check_at is None or stored.next_check_at <= now


def _dominant(values: Iterable[str | None]) -> str | None:
    counts = Counter(value for value in values if value)
    return counts.most_common(1)[0][0] if counts else None


def _epoch(date: datetime | None) -> int:
    return int(date.timestamp()) if date else 0


def _delay(value: float | None) -> float:
    if value is None or not math.isfinite(value) or value < 0:
        return 0.0
    return min(value, MAX_CRAWL_DELAY)


def _retry_after(headers: Mapping[str, str]) -> float:
    try:
        return float(headers.get("Retry-After", 0))
    except ValueError:
        return 0.0


def _gunzip(body: bytes) -> bytes:
    """Decompress gzip up to the protocol's size limit, keeping whatever decoded before a cut."""
    if body[:2] != b"\x1f\x8b":
        return body
    try:
        return zlib.decompressobj(wbits=31).decompress(body, MAX_SITEMAP_BYTES)
    except zlib.error:
        return body
