import pytest

from desearch_bot.urls import HTTPS, TIMED, WWW, Listing, Record, UrlStore, parse


def _key(url, domain="example.com"):
    return parse(url, domain).key


def test_host_case_default_port_and_fragment_do_not_make_a_new_url():
    assert _key("HTTPS://Example.COM:443/Page#top") == _key("https://example.com/Page")


def test_path_case_is_kept_because_servers_treat_it_as_different():
    assert _key("https://example.com/Page") != _key("https://example.com/page")


def test_http_and_https_are_one_page():
    assert _key("http://example.com/a") == _key("https://example.com/a")


def test_www_is_the_same_page_as_the_bare_domain():
    assert _key("https://www.example.com/a") == _key("https://example.com/a")


def test_the_fetchable_address_keeps_the_scheme_and_www_the_site_listed():
    assert (
        parse("https://www.example.com/a", "example.com").fetchable()
        == "https://www.example.com/a"
    )
    assert (
        parse("http://example.com/a", "example.com").fetchable()
        == "http://example.com/a"
    )


def test_query_parameters_are_sorted_and_tracking_ones_dropped():
    url = parse("https://example.com/s?b=2&utm_source=x&a=1&fbclid=z", "example.com")
    assert url.key == _key("https://example.com/s?a=1&b=2")
    assert url.fetchable() == "https://example.com/s?a=1&b=2"


def test_percent_encoding_is_normalised():
    assert _key("https://example.com/caf%c3%a9/%7Euser/a%20b") == _key(
        "https://example.com/caf%C3%A9/~user/a b"
    )


def test_an_empty_path_is_the_root():
    assert (
        parse("https://example.com", "example.com").fetchable()
        == "https://example.com/"
    )


def test_subdomains_belong_to_their_domain_and_other_domains_do_not():
    assert parse("https://blog.example.com/a", "example.com") is not None
    assert parse("https://evil-example.com/a", "example.com") is None
    assert parse("https://example.org/a", "example.com") is None


def test_non_web_and_malformed_addresses_are_refused():
    for bad in (
        "ftp://example.com/a",
        "mailto:someone@example.com",
        "https://",
        "https://example.com:99999/",
        "not a url",
    ):
        assert parse(bad, "example.com") is None


def test_international_hosts_are_stored_in_their_ascii_form():
    assert _key("https://münchen.de/stadt", "münchen.de").startswith(
        b"xn--mnchen-3ya.de\x00"
    )


@pytest.fixture
def store(tmp_path):
    with UrlStore(tmp_path / "urls") as opened:
        yield opened


def _entries(*paths, domain="example.com", lastmod=0, timed=False):
    return [
        (parse(f"https://{domain}{path}", domain), lastmod, timed) for path in paths
    ]


def test_a_first_listing_is_all_new(store):
    assert store.record_listing(1, _entries("/a", "/b", "/c"), now=1000) == Listing(
        3, 3, 0
    )


def test_listing_the_same_urls_again_adds_nothing(store):
    store.record_listing(1, _entries("/a", "/b"), now=1000)
    assert store.record_listing(1, _entries("/a", "/b"), now=2000) == Listing(2, 0, 0)


def test_a_moved_lastmod_is_counted_and_kept(store):
    store.record_listing(1, _entries("/a", lastmod=100), now=1000)
    assert (
        store.record_listing(1, _entries("/a", lastmod=200, timed=True), now=2000).moved
        == 1
    )
    record = store.get(parse("https://example.com/a", "example.com"))
    assert record.lastmod == 200 and record.flags & TIMED
    assert (record.first_seen, record.last_seen) == (1000, 2000)


def test_the_same_page_written_two_ways_is_stored_once(store):
    entries = [
        (parse("https://example.com/a", "example.com"), 0, False),
        (parse("http://www.example.com/a#section", "example.com"), 0, False),
    ]
    assert store.record_listing(1, entries, now=1000).new == 1


def test_a_url_belongs_to_the_sitemap_that_last_listed_it(store):
    store.record_listing(1, _entries("/a"), now=1000)
    store.record_listing(2, _entries("/a"), now=2000)
    assert store.get(parse("https://example.com/a", "example.com")).sitemap_id == 2


def test_a_domains_urls_come_back_together_and_no_others(store):
    store.record_listing(1, _entries("/a", "/b"), now=1000)
    store.record_listing(2, _entries("/z", domain="example.com.au"), now=1000)
    found = [url.fetchable() for url, _ in store.domain("example.com")]
    assert found == ["https://example.com/a", "https://example.com/b"]


def test_urls_survive_closing_and_reopening(tmp_path):
    with UrlStore(tmp_path / "urls") as opened:
        opened.record_listing(1, _entries("/a"), now=1000)
    with UrlStore(tmp_path / "urls") as reopened:
        assert (
            reopened.get(parse("https://example.com/a", "example.com")).first_seen
            == 1000
        )


def test_a_record_packs_and_unpacks_unchanged():
    record = Record(7, 100, 200, 300, 400, 500, HTTPS | WWW)
    assert Record.unpack(record.pack()) == record


def test_simple_and_complicated_hosts_normalise_the_same_way():
    assert _key("https://Example.COM/a") == _key("https://example.com/a")
    assert _key("https://user@example.com/a") == _key("https://example.com/a")
    assert _key("https://example.com:8443/a") != _key("https://example.com/a")
