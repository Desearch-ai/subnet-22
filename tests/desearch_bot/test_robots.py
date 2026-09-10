from desearch_bot.robots import rules, sitemaps


def test_a_rule_for_our_token_wins_over_the_wildcard():
    text = "User-agent: *\nAllow: /\n\nUser-agent: DesearchBot\nDisallow: /\n"
    assert rules(text) == (False, None)


def test_with_no_group_of_our_own_the_wildcard_applies():
    assert rules("User-agent: *\nDisallow: /\n") == (False, None)


def test_a_disallow_on_a_subpath_still_lets_us_in():
    assert rules("User-agent: *\nDisallow: /admin\n")[0] is True


def test_an_empty_file_places_no_restriction():
    assert rules("") == (True, None)


def test_the_crawl_delay_for_our_group_is_returned():
    assert rules("User-agent: DesearchBot\nCrawl-delay: 5\n") == (True, 5.0)


def test_a_nonsense_crawl_delay_is_ignored():
    for value in ("inf", "nan", "-3", "soon"):
        assert rules(f"User-agent: *\nCrawl-delay: {value}\n") == (True, None)


def test_sitemap_lines_are_found_anywhere_in_the_file():
    text = "Sitemap: https://example.com/a.xml\nUser-agent: *\nsitemap:https://example.com/b.xml\n"
    assert sitemaps(text) == ["https://example.com/a.xml", "https://example.com/b.xml"]
