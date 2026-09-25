import asyncio
from datetime import datetime, timezone

import pytest
from aiohttp import web

from desearch.fetch import Fetched, Fetcher, FetchSettings
from neurons.validators.fetchers import (
    OWN_IP_SETTINGS,
    SampleFetcher,
    page_problem,
    to_page,
)
from neurons.validators.scoring import FetchedPage
from tests.local_http import serving
from tests.synthetic import CHALLENGE, synthetic_html

PAGE = synthetic_html(3)
LOOPBACK = FetchSettings(timeout=5, allow_private=True)


def answer(status: int = 200, text: str = PAGE, content_type: str = "text/html"):
    async def handler(_):
        return web.Response(status=status, text=text, content_type=content_type)

    return handler


def fetch(handler, settings: FetchSettings = LOOPBACK) -> FetchedPage:
    async def run():
        async with serving(handler) as base:
            own_ip = Fetcher(settings)
            try:
                return await to_page(await own_ip.fetch(base + "story"))
            finally:
                await own_ip.aclose()

    return asyncio.run(run())


def test_a_page_the_validator_can_reach_is_used_as_is():
    got = fetch(answer())
    assert (got.status, got.error) == (200, "")
    assert page_problem(got, "https://site.example/story") is None


@pytest.mark.parametrize(
    "status, text, error",
    [(403, "<html>no</html>", "http_403"), (503, "<html>later</html>", "http_503")],
)
def test_a_refused_fetch_reports_its_status(status, text, error):
    assert fetch(answer(status, text)).error == error


def test_a_body_over_the_cap_is_refused_without_reading_it_all():
    async def big(_):
        return web.Response(body=b"x" * 5000)

    capped = FetchSettings(timeout=5, max_bytes=1000, allow_private=True)
    assert fetch(big, capped).error == "too_large"


def test_a_challenge_page_is_not_usable():
    blocked = FetchedPage(200, CHALLENGE)
    assert page_problem(blocked, "https://site.example/story") == "blocked"


def test_a_non_html_body_is_not_usable():
    assert page_problem(FetchedPage(200, "%PDF-1.7 binary"), "https://x/") == "not_html"


def through_chain(handler, rendered: bool = False):
    calls = []

    class ScrapingDog:
        async def fetch(self, url, rendered=False, deadline=None):
            calls.append(rendered)
            return Fetched(
                url=url,
                final_url=url,
                fetched_at=datetime.now(timezone.utc),
                status=200,
                body=PAGE.encode(),
            )

    async def run():
        async with serving(handler) as base:
            chain = SampleFetcher(Fetcher(LOOPBACK), ScrapingDog())
            try:
                page, route = await chain.fetch(base + "story", rendered)
            finally:
                await chain.aclose()
            return chain, page, route

    chain, page, route = asyncio.run(run())
    return chain, page, route, calls


def test_scrapingdog_is_not_called_when_the_validator_can_fetch_the_page():
    chain, page, route, calls = through_chain(answer())

    assert calls == [] and route == "own_ip"
    assert (chain.own_ip_fetches, chain.scrapingdog_fetches) == (1, 0)
    assert page.html == PAGE


@pytest.mark.parametrize(
    "handler", [answer(403, "<html>no</html>"), answer(200, CHALLENGE)]
)
def test_scrapingdog_takes_over_when_our_own_ip_is_refused(handler):
    chain, page, route, calls = through_chain(handler)

    assert calls == [False] and route == "scrapingdog"
    assert (chain.own_ip_fetches, chain.scrapingdog_fetches) == (0, 1)
    assert page.html == PAGE


@pytest.mark.parametrize(
    "handler, status",
    [(answer(404, "<html>gone</html>"), 404), (answer(200, "%PDF", "text/plain"), 200)],
)
def test_a_page_fact_is_not_worth_a_scrapingdog_credit(handler, status):
    chain, page, route, calls = through_chain(handler)

    assert calls == [] and route == "own_ip"
    assert page.status == status and page.html == ""


def test_a_rendered_recheck_always_goes_to_scrapingdog():
    chain, _, route, calls = through_chain(answer(), rendered=True)

    assert calls == [True] and route == "scrapingdog"
    assert (chain.own_ip_fetches, chain.scrapingdog_fetches) == (0, 1)


@pytest.mark.parametrize("host", ["127.0.0.1", "localhost"])
def test_the_validator_will_not_fetch_a_private_address(host):
    async def run():
        async with serving(answer()) as base:
            own_ip = Fetcher(OWN_IP_SETTINGS)
            try:
                url = base.replace("127.0.0.1", host) + "admin"
                return await to_page(await own_ip.fetch(url))
            finally:
                await own_ip.aclose()

    got = asyncio.run(run())

    assert (got.html, got.status) == ("", 0)
    assert page_problem(got, "https://site.example/story") is not None


def test_when_scrapingdog_fails_what_our_own_ip_saw_stands():
    class FailingScrapingDog:
        async def fetch(self, url, rendered=False, deadline=None):
            return Fetched(
                url=url,
                final_url=url,
                fetched_at=datetime.now(timezone.utc),
                status=400,
                error="http_4xx",
            )

    async def run():
        async with serving(answer(403, "<html>no</html>")) as base:
            chain = SampleFetcher(Fetcher(LOOPBACK), FailingScrapingDog())
            try:
                return await chain.fetch(base + "story")
            finally:
                await chain.aclose()

    page, route = asyncio.run(run())
    assert (page.status, page.via, route) == (403, "own_ip", "own_ip")


def test_a_javascript_only_stub_is_treated_as_blocked():
    stub = "<html><body>Please enable javascript to access full featured site Reload page</body></html>"
    assert page_problem(FetchedPage(200, stub), "https://site.example/") == "blocked"
