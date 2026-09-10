"""Scripted websites, so the crawler can be exercised without a network."""

import asyncio
import time

ALLOW = b"User-agent: *\nAllow: /\n"
ENGLISH = (
    b"<html lang='en'><body>"
    + b"plenty of readable english words here " * 20
    + b"</body></html>"
)
PAGES = [f"/p{i}" for i in range(12)]


class DNSError(Exception):
    pass


class Response:
    def __init__(self, status, body=b"", headers=None, delay=0.0):
        self.status, self._body, self.headers = status, body, headers or {}
        self.delay = delay

    async def __aenter__(self):
        if self.delay:
            await asyncio.sleep(self.delay)
        return self

    async def __aexit__(self, *_):
        return False

    @property
    def content(self):
        body = self._body

        class Reader:
            async def read(self, limit):
                return body[:limit]

        return Reader()


class FakeWeb:
    """Each address answers with a status and body, an ETag, a redirect, or not at all."""

    def __init__(self):
        self.pages = {}
        self.dead = set()
        self.refused = set()
        self.requested = []
        self.times = []
        self.sent = []

    def page(self, url, body=b"", status=200, etag=None, location=None, delay=0.0):
        self.pages[url] = (status, body, etag, location, delay)

    def get(self, url, headers=None, **_):
        self.requested.append(url)
        self.times.append(time.monotonic())
        self.sent.append(headers or {})
        if url.split("/")[2] in self.dead:
            raise DNSError(url)
        if url in self.refused:
            raise ConnectionRefusedError(url)
        status, body, etag, location, delay = self.pages.get(
            url, (404, b"", None, None, 0.0)
        )
        if etag and headers and headers.get("If-None-Match") == etag:
            return Response(304, delay=delay)
        answer = {"ETag": etag} if etag else {}
        if location:
            answer["Location"] = location
        return Response(status, body, answer, delay)


def urlset(*paths, host="example.com", lastmod=None, news=False, changefreq=None):
    namespace = (
        ' xmlns:news="http://www.google.com/schemas/sitemap-news/0.9"' if news else ""
    )
    stamp = f"<lastmod>{lastmod}</lastmod>" if lastmod else ""
    stamp += f"<changefreq>{changefreq}</changefreq>" if changefreq else ""
    items = "".join(
        f"<url><loc>https://{host}{path}</loc>{stamp}</url>" for path in paths
    )
    return f"<?xml version='1.0'?><urlset{namespace}>{items}</urlset>".encode()


def index(*children):
    items = "".join(
        f"<sitemap><loc>{url}</loc>{f'<lastmod>{date}</lastmod>' if date else ''}</sitemap>"
        for url, date in children
    )
    return f"<?xml version='1.0'?><sitemapindex>{items}</sitemapindex>".encode()


def site(web, robots_txt=ALLOW, sitemap=None, home=ENGLISH, host="example.com"):
    web.page(f"https://{host}/robots.txt", robots_txt)
    if sitemap is not None:
        web.page(f"https://{host}/sitemap.xml", sitemap)
    web.page(f"https://{host}/", home)
