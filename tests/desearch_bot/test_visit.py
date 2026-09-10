import gzip
import time
from datetime import datetime, timedelta, timezone

import pytest

from desearch_bot.schedule import NEWS_INTERVAL, Trust
from desearch_bot.states import Outcome, State
from desearch_bot.urls import UrlStore
from desearch_bot.visit import MAX_FILES, Known, KnownSitemap, Visitor

NOW = datetime(2026, 9, 10, 12, 0, tzinfo=timezone.utc)
ALLOW = b"User-agent: *\nAllow: /\n"
ENGLISH = (
    b"<html lang='en'><body>"
    + b"plenty of readable english words here " * 20
    + b"</body></html>"
)
PAGES = [f"/p{i}" for i in range(12)]


class DNSError(Exception):
    pass


class _Response:
    def __init__(self, status, body=b"", headers=None):
        self.status, self._body, self.headers = status, body, headers or {}

    async def __aenter__(self):
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
    """Scripted sites: each address answers with a status and body, an ETag, or a redirect."""

    def __init__(self):
        self.pages = {}
        self.dead = set()
        self.requested = []
        self.times = []

    def page(self, url, body=b"", status=200, etag=None, location=None):
        self.pages[url] = (status, body, etag, location)

    def get(self, url, headers=None, **_):
        self.requested.append(url)
        self.times.append(time.monotonic())
        if url.split("/")[2] in self.dead:
            raise DNSError(url)
        status, body, etag, location = self.pages.get(url, (404, b"", None, None))
        if etag and headers and headers.get("If-None-Match") == etag:
            return _Response(304)
        answer = {"ETag": etag} if etag else {}
        if location:
            answer["Location"] = location
        return _Response(status, body, answer)


def urlset(*paths, host="example.com", lastmod=None, news=False):
    namespace = (
        ' xmlns:news="http://www.google.com/schemas/sitemap-news/0.9"' if news else ""
    )
    stamp = f"<lastmod>{lastmod}</lastmod>" if lastmod else ""
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


def active(*sitemaps):
    return Known(
        "example.com",
        State.ACTIVE,
        robots_checked_at=NOW - timedelta(hours=1),
        robots_allows=True,
        language="en",
        sitemaps={s.url: s for s in sitemaps},
    )


def known_sitemap(
    url,
    sitemap_id=1,
    kind="urlset",
    depth=0,
    parent_id=None,
    etag=None,
    content_hash=None,
    interval=timedelta(hours=6),
    due=True,
    trust=Trust.UNKNOWN,
    index_lastmod=None,
):
    next_check = NOW - timedelta(minutes=1) if due else NOW + timedelta(hours=1)
    return KnownSitemap(
        sitemap_id,
        url,
        kind,
        depth,
        parent_id,
        etag,
        None,
        content_hash,
        interval,
        next_check,
        trust,
        index_lastmod,
    )


@pytest.fixture
def web():
    return FakeWeb()


@pytest.fixture
def store(tmp_path):
    with UrlStore(tmp_path / "urls") as opened:
        yield opened


@pytest.fixture
def visitor(web, store):
    ids = iter(range(1, 100_000))

    async def allocate(host, url):
        return next(ids)

    return Visitor(
        web,
        store,
        allocate,
        detect_language=lambda text: "en" if "english" in text else "fr",
        registrable=lambda host: host.removeprefix("www."),
        floor=0.0,
    )


async def test_a_new_english_domain_with_a_sitemap_becomes_active_with_its_urls(
    web, visitor
):
    site(web, sitemap=urlset(*PAGES))
    visit = await visitor.visit(Known("example.com"), NOW)
    assert visit.outcome is Outcome.SITEMAP and visit.language == "en"
    assert (visit.listed, visit.new) == (12, 12)


async def test_a_found_sitemap_stops_the_other_guesses(web, visitor):
    site(web, sitemap=urlset(*PAGES))
    visit = await visitor.visit(Known("example.com"), NOW)
    assert "https://example.com/sitemap_index.xml" not in web.requested
    assert [u.url for u in visit.sitemaps] == ["https://example.com/sitemap.xml"]


async def test_sitemaps_named_in_robots_are_followed_through_an_index(web, visitor):
    root, a, b = (
        "https://example.com/sitemap_index.xml",
        "https://example.com/a.xml",
        "https://example.com/b.xml",
    )
    web.page(
        "https://example.com/robots.txt", f"User-agent: *\nSitemap: {root}\n".encode()
    )
    web.page(root, index((a, None), (b, None)))
    web.page(a, urlset(*PAGES[:6]))
    web.page(b, urlset(*PAGES[6:]))
    web.page("https://example.com/", ENGLISH)
    visit = await visitor.visit(Known("example.com"), NOW)
    assert visit.outcome is Outcome.SITEMAP and visit.new == 12
    by_url = {u.url: u for u in visit.sitemaps}
    assert by_url[a].parent_id == by_url[b].parent_id == by_url[root].id


async def test_no_sitemap_anywhere_is_no_sitemap(web, visitor):
    site(web)
    visit = await visitor.visit(Known("example.com"), NOW)
    assert (visit.outcome, visit.reason) == (Outcome.NO_SITEMAP, "no_sitemap")
    assert visit.sitemaps == []


async def test_a_tiny_sitemap_is_not_enough(web, visitor):
    site(web, sitemap=urlset("/a", "/b"))
    visit = await visitor.visit(Known("example.com"), NOW)
    assert (visit.outcome, visit.reason) == (Outcome.NO_SITEMAP, "sitemap_too_small")


async def test_a_site_that_disallows_us_is_blocked_and_nothing_else_is_fetched(
    web, visitor
):
    site(
        web,
        robots_txt=b"User-agent: DesearchBot\nDisallow: /\n",
        sitemap=urlset(*PAGES),
    )
    visit = await visitor.visit(Known("example.com"), NOW)
    assert visit.outcome is Outcome.BLOCKED
    assert web.requested == ["https://example.com/robots.txt"]


async def test_a_domain_that_redirects_away_stops_at_the_first_hop(web, visitor):
    web.page(
        "https://old.com/robots.txt", status=301, location="https://new.com/robots.txt"
    )
    web.page("https://new.com/robots.txt", ALLOW)
    visit = await visitor.visit(Known("old.com"), NOW)
    assert (visit.outcome, visit.canonical_host) == (Outcome.REDIRECT, "new.com")
    assert not any("sitemap" in url for url in web.requested)


async def test_www_is_not_a_redirect_away(web, visitor):
    web.page(
        "https://example.com/robots.txt",
        status=301,
        location="https://www.example.com/robots.txt",
    )
    web.page("https://www.example.com/robots.txt", ALLOW)
    web.page("https://example.com/sitemap.xml", urlset(*PAGES))
    web.page("https://example.com/", ENGLISH)
    visit = await visitor.visit(Known("example.com"), NOW)
    assert visit.outcome is Outcome.SITEMAP and visit.canonical_host is None


async def test_a_domain_with_no_dns_is_unreachable_after_one_attempt(web, visitor):
    web.dead.add("gone.com")
    visit = await visitor.visit(Known("gone.com"), NOW)
    assert visit.outcome is Outcome.UNREACHABLE
    assert web.requested == ["https://gone.com/robots.txt"]


async def test_a_server_error_on_robots_means_we_do_not_crawl_yet(web, visitor):
    web.page("https://example.com/robots.txt", status=503)
    visit = await visitor.visit(Known("example.com"), NOW)
    assert (visit.outcome, visit.reason) == (Outcome.UNREACHABLE, "robots_503")


async def test_a_missing_robots_file_places_no_restriction(web, visitor):
    web.page("https://example.com/sitemap.xml", urlset(*PAGES))
    web.page("https://example.com/", ENGLISH)
    visit = await visitor.visit(Known("example.com"), NOW)
    assert visit.outcome is Outcome.SITEMAP and visit.robots_status == 404


async def test_a_homepage_in_another_language_is_ineligible(web, visitor):
    french = (
        b"<html lang='fr'><body>" + b"bonjour tout le monde " * 30 + b"</body></html>"
    )
    site(web, sitemap=urlset(*PAGES), home=french)
    visit = await visitor.visit(Known("example.com"), NOW)
    assert (visit.outcome, visit.reason) == (Outcome.INELIGIBLE, "not_english")


async def test_a_bot_wall_homepage_is_ineligible(web, visitor):
    wall = b"<html><body>Checking your browser before accessing this site</body></html>"
    site(web, sitemap=urlset(*PAGES), home=wall)
    visit = await visitor.visit(Known("example.com"), NOW)
    assert (visit.outcome, visit.reason) == (Outcome.INELIGIBLE, "bot_wall")


async def test_an_unchanged_sitemap_costs_one_request_and_is_read_less_often(
    web, visitor
):
    url = "https://example.com/sitemap.xml"
    web.page(url, urlset(*PAGES), etag='"v1"')
    visit = await visitor.visit(active(known_sitemap(url, etag='"v1"')), NOW)
    assert web.requested == [url]
    [update] = visit.sitemaps
    assert not update.changed and update.interval > timedelta(hours=6)
    assert visit.outcome is Outcome.SITEMAP


async def test_a_changed_sitemap_adds_only_the_new_urls_and_is_read_more_often(
    web, visitor
):
    url = "https://example.com/sitemap.xml"
    site(web, sitemap=urlset(*PAGES))
    first = await visitor.visit(Known("example.com"), NOW)
    web.page(url, urlset(*PAGES, "/p99"))
    second = await visitor.visit(
        active(known_sitemap(url, content_hash=first.sitemaps[0].content_hash)), NOW
    )
    assert (second.listed, second.new) == (13, 1)
    [update] = second.sitemaps
    assert update.changed and update.interval < timedelta(hours=6)


async def test_a_sitemap_that_is_not_yet_due_is_left_alone(web, visitor):
    url = "https://example.com/sitemap.xml"
    web.page(url, urlset(*PAGES))
    visit = await visitor.visit(active(known_sitemap(url, due=False)), NOW)
    assert web.requested == [] and visit.sitemaps == []


async def test_a_trusted_index_only_sends_us_to_the_children_whose_date_moved(
    web, visitor
):
    root, a, b = (
        "https://example.com/sitemap_index.xml",
        "https://example.com/a.xml",
        "https://example.com/b.xml",
    )
    web.page(root, index((a, "2026-09-01"), (b, "2026-09-09")))
    web.page(a, urlset(*PAGES[:6]))
    web.page(b, urlset(*PAGES[6:]))
    await visitor.visit(
        active(
            known_sitemap(
                root, 1, kind="index", content_hash="old", trust=Trust.TRUSTED
            ),
            known_sitemap(
                a, 2, depth=1, parent_id=1, due=False, index_lastmod="2026-09-01"
            ),
            known_sitemap(
                b, 3, depth=1, parent_id=1, due=False, index_lastmod="2026-09-01"
            ),
        ),
        NOW,
    )
    assert web.requested == [root, b]


async def test_one_visit_reads_at_most_the_file_limit(web, visitor):
    children = [f"https://example.com/s{i}.xml" for i in range(MAX_FILES + 10)]
    web.page(
        "https://example.com/robots.txt", b"Sitemap: https://example.com/idx.xml\n"
    )
    web.page(
        "https://example.com/idx.xml", index(*[(child, None) for child in children])
    )
    for child in children:
        web.page(child, urlset(*PAGES))
    web.page("https://example.com/", ENGLISH)
    await visitor.visit(Known("example.com"), NOW)
    assert len([url for url in web.requested if url.endswith(".xml")]) == MAX_FILES


async def test_a_news_sitemap_starts_on_the_news_schedule(web, visitor):
    site(web, sitemap=urlset(*PAGES, news=True))
    visit = await visitor.visit(Known("example.com"), NOW)
    assert visit.sitemaps[0].interval == NEWS_INTERVAL


async def test_a_gzipped_sitemap_is_read(web, visitor):
    site(web, sitemap=gzip.compress(urlset(*PAGES)))
    visit = await visitor.visit(Known("example.com"), NOW)
    assert visit.new == 12


async def test_urls_on_other_domains_are_not_taken(web, visitor):
    stray = b"<url><loc>https://elsewhere.com/x</loc></url></urlset>"
    site(web, sitemap=urlset(*PAGES).replace(b"</urlset>", stray))
    visit = await visitor.visit(Known("example.com"), NOW)
    assert visit.listed == 12


async def test_the_crawl_delay_in_robots_spaces_every_later_request(web, visitor):
    site(web, robots_txt=b"User-agent: *\nCrawl-delay: 0.3\n", sitemap=urlset(*PAGES))
    visit = await visitor.visit(Known("example.com"), NOW)
    assert visit.crawl_delay == 0.3
    gaps = [later - earlier for earlier, later in zip(web.times, web.times[1:])]
    assert gaps and all(gap >= 0.28 for gap in gaps)
