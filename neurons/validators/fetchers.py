from __future__ import annotations

import asyncio
import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor

import aiohttp

from desearch.extraction import extract, looks_blocked
from desearch.fetch import (
    Fetched,
    Fetcher,
    FetchSettings,
    ScrapingDog,
    decode_html,
    needs_fallback,
)
from neurons.validators.scoring import (
    OWN_IP,
    SCRAPINGDOG,
    FetchedPage,
    fetch_failed,
    looks_like_html,
    provider_refused,
)

# One try from our own address; a page that does not answer in time goes to ScrapingDog.
OWN_IP_SETTINGS = FetchSettings(concurrency=256, timeout=8.0, max_bytes=5_000_000)
EXTRACTION_THREADS = 4
SCRAPINGDOG_ACCOUNT_URL = "https://api.scrapingdog.com/account"
ACCOUNT_TIMEOUT = aiohttp.ClientTimeout(total=30.0)
RUNNING_OUT_S = 86400.0

log = logging.getLogger("validator")


async def to_page(fetched: Fetched, via: str = "") -> FetchedPage:
    if fetched.status and fetched.status != 200:
        return FetchedPage(fetched.status, error=f"http_{fetched.status}", via=via)
    if fetched.body is None:
        return FetchedPage(fetched.status, error=fetched.error or "other", via=via)
    html = await asyncio.to_thread(decode_html, fetched.body, fetched.charset)
    return FetchedPage(200, html, via=via)


def page_problem(page: FetchedPage, url: str) -> str | None:
    if page.error:
        return page.error
    if page.via != OWN_IP and not looks_like_html(page.html):
        return "not_html"
    if looks_blocked(page.status, page.html, extract(page.html, url).text):
        return "blocked"
    return None


class HourlyBudget:
    def __init__(self, per_hour: int):
        self.per_hour = per_hour
        self.hour = -1
        self.used = 0

    def take(self) -> bool:
        hour = int(time.time() // 3600)
        if hour != self.hour:
            self.hour, self.used = hour, 0
        if self.used >= self.per_hour:
            return False
        self.used += 1
        if self.used == self.per_hour:
            log.warning(
                "ScrapingDog's %d requests for this hour are used; until the next hour"
                " pages our own IP cannot load are judged without it",
                self.per_hour,
            )
        return True


class SampleFetcher:
    def __init__(
        self,
        own_ip: Fetcher,
        scrapingdog: ScrapingDog,
        budget: HourlyBudget | None = None,
    ):
        self.own_ip = own_ip
        self.scrapingdog = scrapingdog
        self.budget = budget
        self.own_ip_fetches = 0
        self.scrapingdog_fetches = 0
        self.extraction = ThreadPoolExecutor(
            EXTRACTION_THREADS, thread_name_prefix="extract"
        )

    async def __call__(self, url: str) -> FetchedPage:
        return (await self.fetch(url))[0]

    async def fetch(self, url: str) -> tuple[FetchedPage, str]:
        """ScrapingDog can add evidence but never erase what our own IP saw."""
        own = await to_page(await self.own_ip.attempt(url), OWN_IP)
        problem = await asyncio.get_running_loop().run_in_executor(
            self.extraction, page_problem, own, url
        )
        if problem is None or not needs_fallback(problem, own.status):
            self.own_ip_fetches += 1
            return own, OWN_IP
        if self.budget is not None and not self.budget.take():
            return own, OWN_IP
        self.scrapingdog_fetches += 1
        page = await to_page(await self.scrapingdog.fetch(url), SCRAPINGDOG)
        if fetch_failed(page) and not provider_refused(page):
            return own, OWN_IP
        return page, SCRAPINGDOG

    async def aclose(self) -> None:
        await self.own_ip.aclose()
        self.extraction.shutdown(wait=False, cancel_futures=True)


class ScrapingDogAccount:
    def __init__(
        self,
        http: aiohttp.ClientSession,
        api_key: str,
        endpoint: str = SCRAPINGDOG_ACCOUNT_URL,
    ):
        self.http = http
        self.api_key = api_key
        self.endpoint = endpoint
        self.last_read: tuple[float, int] | None = None

    async def problem(self) -> str | None:
        async with self.http.get(
            self.endpoint, params={"api_key": self.api_key}, timeout=ACCOUNT_TIMEOUT
        ) as response:
            status, body = response.status, await response.text()
        if status >= 500 or status == 429:
            raise RuntimeError(f"ScrapingDog account endpoint answered HTTP {status}")
        if status != 200:
            return f"ScrapingDog refuses the key: HTTP {status} {body[:200]}"
        account = json.loads(body)
        if account.get("dailyCreditsRemaining") == 0:
            return "ScrapingDog's daily credit limit is used up"
        limit, used = account.get("requestLimit"), account.get("requestUsed")
        if limit is None or used is None:
            return None
        if used >= limit:
            return f"ScrapingDog has no credits left: {used:,} of {limit:,} used"
        self.note_pace(used, limit - used)
        return None

    def note_pace(self, used: int, left: int) -> None:
        now = time.monotonic()
        if self.last_read is not None:
            then, used_then = self.last_read
            pace = (used - used_then) / (now - then) if now > then else 0.0
            if pace > 0 and left / pace < RUNNING_OUT_S:
                log.warning(
                    "ScrapingDog has %s credits left, about %.1f hours at the current pace",
                    f"{left:,}",
                    left / pace / 3600,
                )
        self.last_read = (now, used)
