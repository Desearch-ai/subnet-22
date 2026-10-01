from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor

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
)

OWN_IP_SETTINGS = FetchSettings(timeout=20.0, max_bytes=5_000_000)
EXTRACTION_THREADS = 4


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


class SampleFetcher:
    def __init__(self, own_ip: Fetcher, scrapingdog: ScrapingDog):
        self.own_ip = own_ip
        self.scrapingdog = scrapingdog
        self.own_ip_fetches = 0
        self.scrapingdog_fetches = 0
        self.extraction = ThreadPoolExecutor(
            EXTRACTION_THREADS, thread_name_prefix="extract"
        )

    async def __call__(self, url: str, rendered: bool = False) -> FetchedPage:
        return (await self.fetch(url, rendered))[0]

    async def fetch(self, url: str, rendered: bool = False) -> tuple[FetchedPage, str]:
        """ScrapingDog can add evidence but never erase what our own IP saw."""
        own = None
        if not rendered:
            own = await to_page(await self.own_ip.fetch(url), OWN_IP)
            problem = await asyncio.get_running_loop().run_in_executor(
                self.extraction, page_problem, own, url
            )
            if problem is None or not needs_fallback(problem, own.status):
                self.own_ip_fetches += 1
                return own, OWN_IP
        self.scrapingdog_fetches += 1
        page = await to_page(await self.scrapingdog.fetch(url, rendered), SCRAPINGDOG)
        if own is not None and fetch_failed(page):
            return own, OWN_IP
        return page, SCRAPINGDOG

    async def aclose(self) -> None:
        await self.own_ip.aclose()
        self.extraction.shutdown(wait=False, cancel_futures=True)
