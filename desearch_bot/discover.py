"""Qualify a domain and collect the URLs its sitemaps list, in one visit.

Sitemap rows and their refresh schedule go to Postgres; the URLs go to the frontier.
"""

from __future__ import annotations

import asyncio
import hashlib
import time

import aiohttp

from . import db, sitemaps
from .frontier import Frontier
from .qualify import (
    MAX_SITEMAP_BYTES,
    MIN_HOST_INTERVAL,
    Qualifier,
    Result,
    _gunzip,
    _resolver,
)
from .sources import USER_AGENT

MAX_SITEMAP_FILES = 40
MAX_DEPTH = 3

# How often to recheck a sitemap, before the adaptive schedule takes over.
INTERVAL_BY_CHANGEFREQ = {
    "always": 1800,
    "hourly": 3600,
    "daily": 21600,
    "weekly": 172800,
    "monthly": 604800,
    "yearly": 2592000,
    "never": 2592000,
}
DEFAULT_INTERVAL = 86400
NEWS_INTERVAL = 3600


def initial_interval(kind: str | None, changefreq: str | None, url: str) -> int:
    if "news" in url.lower():
        return NEWS_INTERVAL
    if changefreq in INTERVAL_BY_CHANGEFREQ:
        return INTERVAL_BY_CHANGEFREQ[changefreq]
    return DEFAULT_INTERVAL


def dominant_changefreq(urls) -> str | None:
    values = [changefreq for *_, changefreq in urls if changefreq]
    return max(set(values), key=values.count) if values else None


class Walker:
    """Walks one domain's sitemap tree, breadth-first, storing what it finds."""

    def __init__(
        self, session: aiohttp.ClientSession, pool, frontier: Frontier, timeout: float
    ):
        self.session = session
        self.pool = pool
        self.frontier = frontier
        self.timeout = aiohttp.ClientTimeout(total=timeout, connect=min(timeout, 6.0))
        self.crawl_delay = 0.0
        self._next_request_at = 0.0

    async def fetch(self, url: str):
        wait = self._next_request_at - time.monotonic()
        if wait > 0:
            await asyncio.sleep(wait)
        self._next_request_at = time.monotonic() + max(self.crawl_delay, MIN_HOST_INTERVAL)
        async with self.session.get(
            url,
            timeout=self.timeout,
            allow_redirects=True,
            headers={"User-Agent": USER_AGENT},
        ) as response:
            body = await response.content.read(MAX_SITEMAP_BYTES)
            return response.status, body, response.headers

    async def walk(self, result: Result) -> int:
        self.crawl_delay = result.crawl_delay or 0.0
        self._next_request_at = time.monotonic() + max(self.crawl_delay, MIN_HOST_INTERVAL)
        queue = [(result.sitemap_url, 0, None)]
        seen = {result.sitemap_url}
        stored = fetches = 0

        while queue and fetches < MAX_SITEMAP_FILES:
            url, depth, parent_id = queue.pop(0)
            fetches += 1
            try:
                status, body, headers = await self.fetch(url)
            except Exception as exc:
                await db.save_sitemap(
                    self.pool,
                    result.host,
                    url,
                    None,
                    depth,
                    parent_id=parent_id,
                    status="error",
                    error=type(exc).__name__,
                )
                continue
            if status != 200 or not body:
                await db.save_sitemap(
                    self.pool,
                    result.host,
                    url,
                    None,
                    depth,
                    parent_id=parent_id,
                    status="error",
                    error=f"http {status}",
                )
                continue

            payload = _gunzip(body)
            kind, entries = sitemaps.parse_entries(payload)
            walk = sitemaps.collect(kind, entries, 1 << 30)
            changefreq = dominant_changefreq(walk.urls)

            sitemap_id = await db.save_sitemap(
                self.pool,
                result.host,
                url,
                kind,
                depth,
                url_count=len(walk.urls),
                child_count=len(walk.children),
                parent_id=parent_id,
                etag=headers.get("ETag"),
                last_modified=headers.get("Last-Modified"),
                content_hash=hashlib.sha256(payload).hexdigest()[:32],
                check_interval_s=initial_interval(kind, changefreq, url),
            )
            if walk.urls:
                stored += self.frontier.add(result.host, sitemap_id, walk.urls)
            if walk.children and depth < MAX_DEPTH:
                for child in walk.children:
                    if child.url not in seen:
                        seen.add(child.url)
                        queue.append((child.url, depth + 1, sitemap_id))
        return stored


async def discover(
    pool,
    frontier: Frontier,
    hosts,
    detect_language,
    concurrency: int = 150,
    timeout: float = 8.0,
    on_done=None,
    adult: set[str] = frozenset(),
) -> None:
    queue: asyncio.Queue = asyncio.Queue(maxsize=concurrency * 4)
    connector = aiohttp.TCPConnector(
        limit=concurrency,
        limit_per_host=4,
        ttl_dns_cache=900,
        enable_cleanup_closed=True,
        resolver=_resolver(),
    )
    pending: list[tuple[Result, int]] = []
    lock = asyncio.Lock()

    async def flush(force: bool = False) -> None:
        async with lock:
            if not pending or (len(pending) < 200 and not force):
                return
            batch, pending[:] = list(pending), []
        await db.save_domains(pool, batch)

    async with aiohttp.ClientSession(connector=connector) as session:
        qualifier = Qualifier(session, detect_language, timeout, adult)
        walker = Walker(session, pool, frontier, timeout)

        async def worker():
            while True:
                entry = await queue.get()
                if entry is None:
                    queue.task_done()
                    return
                try:
                    result = await qualifier.run(*entry)
                    urls = await walker.walk(result) if result.qualified else 0
                except Exception as exc:
                    result = Result(
                        host=entry[0],
                        reject_reason=f"crashed: {type(exc).__name__}",
                        error=str(exc)[:200],
                    )
                    urls = 0
                pending.append((result, urls))
                await flush()
                if on_done:
                    on_done(result, urls)
                queue.task_done()

        workers = [asyncio.create_task(worker()) for _ in range(concurrency)]
        for entry in hosts:
            await queue.put(entry)
        for _ in workers:
            await queue.put(None)
        await asyncio.gather(*workers)

    await flush(force=True)
    frontier.flush()


class Progress:
    def __init__(self, every: int = 500):
        self.every = every
        self.checked = self.qualified = self.urls = 0
        self.start = time.monotonic()

    def __call__(self, result: Result, urls: int) -> None:
        self.checked += 1
        self.qualified += bool(result.qualified)
        self.urls += urls
        if self.checked % self.every == 0:
            rate = self.checked / max(time.monotonic() - self.start, 1)
            print(
                f"  {self.checked:,} checked  {self.qualified:,} qualified  "
                f"{self.urls:,} urls  {rate:.0f}/s",
                flush=True,
            )
