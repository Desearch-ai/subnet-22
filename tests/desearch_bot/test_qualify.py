import asyncio
import time

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from desearch_bot import signing
from desearch_bot.qualify import Pacer, Qualifier, parse_sitemap, robots_allows
from desearch_bot.suffixes import ENGLISH_MARKET, tld_group


def test_wildcard_group_applies_when_token_absent():
    assert robots_allows("User-agent: *\nDisallow: /", "DesearchBot") == (False, None)
    assert robots_allows("User-agent: *\nDisallow:", "DesearchBot") == (True, None)


def test_named_group_overrides_wildcard():
    text = "User-agent: *\nDisallow: /\n\nUser-agent: DesearchBot\nDisallow:\nCrawl-delay: 5"
    assert robots_allows(text, "DesearchBot") == (True, 5.0)


def test_disallowing_a_subpath_leaves_the_root_crawlable():
    assert robots_allows("User-agent: *\nDisallow: /private/", "DesearchBot")[0] is True


def test_grouped_agents_share_rules():
    text = "User-agent: BadBot\nUser-agent: DesearchBot\nDisallow: /"
    assert robots_allows(text, "DesearchBot")[0] is False


def test_comments_and_blank_lines_are_ignored():
    text = "# comment\n\nUser-agent: *  # trailing\nDisallow: /"
    assert robots_allows(text, "DesearchBot")[0] is False


def test_missing_robots_rules_allow_crawling():
    assert robots_allows("", "DesearchBot") == (True, None)


def test_urlset_reports_page_count():
    body = b"<urlset><url><loc>https://a.test/1</loc></url><url><loc>https://a.test/2</loc></url></urlset>"
    kind, count, sample = parse_sitemap(body)
    assert (kind, count) == ("urlset", 2)
    assert sample == ["https://a.test/1", "https://a.test/2"]


def test_index_is_distinguished_from_urlset():
    body = b"<sitemapindex><sitemap><loc>https://a.test/s.xml</loc></sitemap></sitemapindex>"
    assert parse_sitemap(body)[0] == "index"


def test_gzipped_sitemap_is_decompressed():
    import gzip

    body = gzip.compress(b"<urlset><url><loc>https://a.test/1</loc></url></urlset>")
    assert parse_sitemap(body)[:2] == ("urlset", 1)


def test_html_is_not_a_sitemap():
    assert parse_sitemap(b"<html><body>not a sitemap</body></html>")[0] == "invalid"


def test_tld_groups():
    assert tld_group("example.com") == "big_generic"
    assert tld_group("example.co.uk") == "en_cctld"
    assert tld_group("example.io") == "new_generic"
    assert tld_group("example.de") == "other_cctld"
    assert "co.uk" in ENGLISH_MARKET


class _Response:
    status, headers = 200, {}

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_):
        return False

    @property
    def content(self):
        class Reader:
            async def read(self, _):
                return b"<urlset></urlset>"

        return Reader()


class _Session:
    def __init__(self):
        self.log = []

    def get(self, url, **kwargs):
        self.log.append((time.monotonic(), url.split("/")[2], kwargs["headers"]))
        return _Response()


async def _visit(qualifier, host, delay, start_after):
    await asyncio.sleep(start_after)
    pacer = Pacer(delay)
    for path in ("/robots.txt", "/sitemap.xml", "/"):
        await qualifier._get(f"https://{host}{path}", 10, pacer)


async def test_hosts_are_paced_on_their_own_clocks():
    """A slow host must not slow its neighbours, and must not be sped up by them."""
    session = _Session()
    qualifier = Qualifier(session, lambda _: "en", 5.0)
    await asyncio.gather(
        _visit(qualifier, "slow.example", 2.0, 0.0),
        _visit(qualifier, "fast.example", 0.0, 0.3),
        _visit(qualifier, "other.example", 0.0, 0.7),
    )

    times = {}
    for at, host, _ in session.log:
        times.setdefault(host, []).append(at)

    for host, needed in (("slow.example", 2.0), ("fast.example", 1.0), ("other.example", 1.0)):
        gaps = [b - a for a, b in zip(times[host], times[host][1:])]
        assert all(gap >= needed - 0.05 for gap in gaps), (host, gaps)
        assert all(gap < needed + 0.5 for gap in gaps), (host, gaps)


async def test_requests_carry_the_signature_when_a_key_is_configured():
    session = _Session()
    signer = signing.Signer(Ed25519PrivateKey.generate())
    qualifier = Qualifier(session, lambda _: "en", 5.0, signer=signer)
    await qualifier._get("https://example.com/robots.txt", 10, Pacer())

    _, _, headers = session.log[0]
    assert headers["Signature-Agent"] == '"https://www.desearch.ai/crawler"'
    assert f'keyid="{signer.keyid}"' in headers["Signature-Input"]


async def test_requests_are_unsigned_when_no_key_is_configured():
    session = _Session()
    qualifier = Qualifier(session, lambda _: "en", 5.0)
    await qualifier._get("https://example.com/robots.txt", 10, Pacer())
    assert "Signature" not in session.log[0][2]


async def test_crawl_delay_applies_to_the_request_right_after_robots():
    """robots.txt is fetched before its Crawl-delay is known; the next request must still wait."""
    pacer = Pacer()
    await pacer.wait()
    pacer.slow_to(2.0)
    start = time.monotonic()
    await pacer.wait()
    assert time.monotonic() - start >= 1.95


async def test_slow_to_never_shortens_an_interval():
    pacer = Pacer(5.0)
    pacer.slow_to(1.0)
    assert pacer.interval == 5.0
