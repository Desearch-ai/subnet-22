import asyncio

from aiohttp import web

from desearch.fetch import ScrapingDog
from neurons.validators.scoring import (
    FetchedPage,
    cleared_by_render,
    needs_rendered_check,
)
from tests.local_http import serving
from tests.synthetic import error_row, page_row, synthetic_html, synthetic_url

THIN = "<html><head><title>Loading</title></head><body>Please wait</body></html>"


def recheck(kept, fetched, pages):
    calls = []

    async def fetch(url: str, rendered: bool = False) -> FetchedPage:
        calls.append((url, rendered))
        return pages[url]

    async def run():
        rendered = {
            url: await fetch(url, rendered=True)
            for url in needs_rendered_check(kept, fetched)
        }
        return cleared_by_render(kept, rendered)

    return asyncio.run(run()), calls


def test_a_thin_first_fetch_is_replaced_by_the_rendered_one():
    url = synthetic_url(1)
    real = FetchedPage(200, synthetic_html(1))
    better, calls = recheck(
        {url: page_row(url, synthetic_html(1))},
        {url: FetchedPage(200, THIN)},
        {url: real},
    )
    assert better == {url: real}
    assert calls == [(url, True)]


def test_a_failed_first_fetch_is_replaced_by_the_rendered_one():
    url = synthetic_url(2)
    real = FetchedPage(200, synthetic_html(2))
    better, _ = recheck(
        {url: page_row(url, synthetic_html(2))},
        {url: FetchedPage(400, error="http_400")},
        {url: real},
    )
    assert better == {url: real}


def test_forged_text_stays_mismatched():
    url = synthetic_url(3)
    forged = page_row(url, synthetic_html(99))
    real = FetchedPage(200, synthetic_html(3))
    better, calls = recheck({url: forged}, {url: real}, {url: real})
    assert better == {}
    assert calls == [(url, True)]


def test_matched_and_error_rows_are_not_refetched():
    ok, failed = synthetic_url(4), synthetic_url(5)
    kept = {ok: page_row(ok, synthetic_html(4)), failed: error_row(failed)}
    fetched = {
        ok: FetchedPage(200, synthetic_html(4)),
        failed: FetchedPage(200, synthetic_html(5)),
    }
    better, calls = recheck(kept, fetched, {})
    assert better == {}
    assert calls == []


def test_a_worse_rendered_fetch_does_not_replace_a_mismatch():
    url = synthetic_url(6)
    first = FetchedPage(200, synthetic_html(7))
    better, _ = recheck(
        {url: page_row(url, synthetic_html(6))},
        {url: first},
        {url: FetchedPage(0, error="timeout")},
    )
    assert better == {}


def test_scrapingdog_retries_its_own_400_and_renders_on_request():
    calls = []
    plan = {"https://a.example/": [400, 200], "https://b.example/": [200]}

    async def handler(request: web.Request) -> web.Response:
        calls.append(request.query["dynamic"])
        return web.Response(
            status=plan[request.query["url"]].pop(0), text="<html></html>"
        )

    async def fetch_all():
        async with (
            serving(handler) as endpoint,
            ScrapingDog("k", endpoint=endpoint) as scrapingdog,
        ):
            return await scrapingdog.fetch(
                "https://a.example/"
            ), await scrapingdog.fetch("https://b.example/", rendered=True)

    retried, rendered = asyncio.run(fetch_all())
    assert (retried.status, retried.error, retried.attempts) == (200, None, 2)
    assert rendered.status == 200
    assert calls == ["false", "false", "true"]


def test_a_rendered_fetch_cannot_turn_an_unchecked_sample_into_a_mismatch():
    url = synthetic_url(8)
    shell = (
        "<html><head><title>Shop</title></head><body><div id=app></div></body></html>"
    )
    rendered = FetchedPage(200, synthetic_html(8))
    better, calls = recheck(
        {url: page_row(url, shell)},
        {url: FetchedPage(400, error="http_400")},
        {url: rendered},
    )
    assert calls == [(url, True)]
    assert better == {}
