from desearch_bot.qualify import parse_sitemap, robots_allows
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
