"""Which domains are reachable, and which name is the real one.

DNS runs first because it is nearly free and removes most of the dead. The survivors then get one
request each, following redirects by hand: a domain that redirects to another is not a crawl
target of its own, and treating it as one would mean hitting the same server twice under two
names, each believing it owns the full per-host request budget.

Lookups go to a recursive resolver on localhost. Public resolvers rate-limit a bulk pass, and a
refusal is indistinguishable from a domain that does not exist.
"""

from __future__ import annotations

import asyncio
import time
from urllib.parse import urljoin, urlsplit

import aiohttp

from . import db, signing
from .qualify import MAX_REDIRECTS, REDIRECT_STATUSES, Pacer
from .suffixes import PublicSuffixList

RESOLVER_HOST = "127.0.0.1"
RESOLVER_PORT = 5335

# A domain that exists but publishes no address is as dead to a crawler as one that does not
# exist; both are recorded as not resolving.
DEAD = {"NXDOMAIN", "NODATA", "SERVFAIL"}


def _resolver():
    import aiodns

    return aiodns.DNSResolver(
        nameservers=[RESOLVER_HOST],
        udp_port=RESOLVER_PORT,
        tcp_port=RESOLVER_PORT,
        timeout=5.0,
        tries=2,
    )


async def resolve_all(pool, concurrency: int = 200, rate: float = 300.0,
                      on_batch=None) -> dict:
    """Walks every unchecked domain. Rate matters more than concurrency: a recursive lookup is
    several small packets each way, and a few thousand a second is enough to saturate the packet
    budget of a virtual NIC and starve everything else on the box, SSH included."""
    resolver = _resolver()
    semaphore = asyncio.Semaphore(concurrency)
    counts = {"checked": 0, "resolves": 0, "dead": 0}
    interval = 1.0 / rate
    started = time.monotonic()
    issued = 0

    async def slot() -> None:
        """Each lookup waits for its own place in the schedule. Holding a lock across the sleep
        would serialise every caller on one wakeup and cap throughput far below the rate."""
        nonlocal issued
        issued += 1
        delay = started + issued * interval - time.monotonic()
        if delay > 0:
            await asyncio.sleep(delay)

    async def check(host: str) -> tuple[str, bool]:
        await slot()
        async with semaphore:
            try:
                await resolver.query_dns(host, "A")
                return host, True
            except Exception:
                return host, False

    async for rows in db.iter_unresolved_hosts(pool):
        hosts = [r["host"] for r in rows]
        results = await asyncio.gather(*(check(h) for h in hosts))
        await db.save_resolution(pool, results)
        counts["checked"] += len(results)
        counts["resolves"] += sum(1 for _, ok in results if ok)
        counts["dead"] += sum(1 for _, ok in results if not ok)
        if on_batch:
            on_batch(counts)
    return counts


class Canonicaliser:
    """Follows redirects to find the name a domain actually serves under."""

    def __init__(self, session: aiohttp.ClientSession, psl: PublicSuffixList, signer=None,
                 timeout: float = 10.0):
        self.session = session
        self.psl = psl
        self.signer = signer
        self.timeout = aiohttp.ClientTimeout(total=timeout, connect=6.0, sock_connect=6.0)

    async def final_host(self, host: str) -> tuple[str, str | None, str | None]:
        """Returns (host, canonical_host, error). canonical_host is None when the domain serves
        under its own name, which is the common case."""
        pacer = Pacer()
        for scheme in ("https", "http"):
            url = f"{scheme}://{host}/"
            try:
                landed = await self._follow(url, pacer)
            except Exception as exc:
                error = type(exc).__name__
                continue
            registrable = self.psl.registrable(urlsplit(landed).hostname or "")
            if not registrable or registrable == host:
                return host, None, None
            return host, registrable, None
        return host, None, error

    async def _follow(self, url: str, pacer: Pacer) -> str:
        for _ in range(MAX_REDIRECTS + 1):
            await pacer.wait()
            async with self.session.get(
                url,
                timeout=self.timeout,
                allow_redirects=False,
                headers=signing.request_headers(url, self.signer),
            ) as response:
                location = response.headers.get("Location")
                if response.status in REDIRECT_STATUSES and location:
                    url = urljoin(url, location)
                    continue
                await response.release()
                return url
        raise RuntimeError("TooManyRedirects")


# A session accumulates per-host state -- TLS contexts, connection-pool entries, resolver cache
# -- and this pass visits millions of distinct hosts. Replacing it periodically releases that;
# without it the process grew to 6 GB over twelve hours.
BATCHES_PER_SESSION = 40


def _session(concurrency: int) -> aiohttp.ClientSession:
    return aiohttp.ClientSession(
        connector=aiohttp.TCPConnector(
            limit=concurrency,
            limit_per_host=2,
            ttl_dns_cache=300,
            enable_cleanup_closed=True,
            force_close=True,
            resolver=aiohttp.resolver.AsyncResolver(
                nameservers=[RESOLVER_HOST], udp_port=RESOLVER_PORT, tcp_port=RESOLVER_PORT
            ),
        )
    )


async def canonicalise_all(pool, psl: PublicSuffixList, concurrency: int = 200,
                           signer=None, on_batch=None) -> dict:
    counts = {"checked": 0, "redirects": 0, "errors": 0}
    semaphore = asyncio.Semaphore(concurrency)
    session = _session(concurrency)
    batches = 0

    try:
        async for rows in db.iter_resolving_hosts(pool):
            worker = Canonicaliser(session, psl, signer)

            async def one(host):
                async with semaphore:
                    return await worker.final_host(host)

            hosts = [r["host"] for r in rows]
            results = await asyncio.gather(*(one(h) for h in hosts))
            await db.save_canonical(pool, [(h, c, e is None) for h, c, e in results])
            counts["checked"] += len(results)
            counts["redirects"] += sum(1 for _, c, _ in results if c)
            counts["errors"] += sum(1 for _, c, e in results if e and not c)
            if on_batch:
                on_batch(counts)

            batches += 1
            if batches % BATCHES_PER_SESSION == 0:
                await session.close()
                session = _session(concurrency)
    finally:
        await session.close()
    return counts
