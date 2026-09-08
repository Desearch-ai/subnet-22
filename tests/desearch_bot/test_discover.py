import asyncio
import time

from desearch_bot import db, discover
from desearch_bot.qualify import Fetched, Pacer, Result


class _Response:
    def __init__(self, headers):
        self.status, self.headers = 200, headers

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_):
        return False

    @property
    def content(self):
        class Reader:
            async def read(self, _):
                return b"<?xml version='1.0'?><urlset></urlset>"

        return Reader()


class _Session:
    def __init__(self):
        self.fetched = []

    def get(self, url, **kwargs):
        self.fetched.append(url)
        return _Response({"ETag": '"child"'})


class _Frontier:
    def add(self, host, sitemap_id, urls):
        return len(urls)


ROOT = "https://news-publisher.com/sitemap.xml"
CHILD = "https://news-publisher.com/posts.xml"
INDEX = (
    "<?xml version='1.0'?><sitemapindex><sitemap>"
    f"<loc>{CHILD}</loc>"
    "</sitemap></sitemapindex>"
).encode()


def _qualified():
    return Result(
        host="news-publisher.com",
        qualified=True,
        sitemap_url=ROOT,
        sitemap_kind="index",
        sitemap_fetch=Fetched(INDEX, {"ETag": '"root"', "Last-Modified": "Tue, 08 Sep 2026 10:00:00 GMT"}),
    )


async def _walk(result):
    saved = []

    async def save_sitemap(pool, host, url, kind, depth, **kwargs):
        saved.append((url, kwargs.get("etag"), kwargs.get("last_modified")))
        return len(saved)

    original, db.save_sitemap = db.save_sitemap, save_sitemap
    try:
        session = _Session()
        walker = discover.Walker(session, None, _Frontier(), 5.0)
        await walker.walk(result, Pacer())
        return session.fetched, saved
    finally:
        db.save_sitemap = original


async def test_the_root_sitemap_is_not_fetched_a_second_time():
    fetched, saved = await _walk(_qualified())
    assert ROOT not in fetched
    assert fetched == [CHILD]
    assert [url for url, _, _ in saved] == [ROOT, CHILD]


async def test_the_headers_from_qualification_are_kept_for_the_root():
    _, saved = await _walk(_qualified())
    root_url, etag, last_modified = saved[0]
    assert root_url == ROOT
    assert etag == '"root"'
    assert last_modified == "Tue, 08 Sep 2026 10:00:00 GMT"


async def test_a_result_without_a_prefetch_still_walks():
    result = _qualified()
    result.sitemap_fetch = None
    fetched, _ = await _walk(result)
    assert fetched[0] == ROOT
