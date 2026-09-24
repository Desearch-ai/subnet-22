from desearch_bot.urls import HTTPS, WWW, Record, parse


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


def test_a_record_packs_and_unpacks_unchanged():
    record = Record(7, 100, 200, 300, 400, 500, HTTPS | WWW)
    assert Record.unpack(record.pack()) == record


def test_simple_and_complicated_hosts_normalise_the_same_way():
    assert _key("https://Example.COM/a") == _key("https://example.com/a")
    assert _key("https://user@example.com/a") == _key("https://example.com/a")
    assert _key("https://example.com:8443/a") != _key("https://example.com/a")
