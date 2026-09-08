from desearch_bot.exclusions import exclusion_reason
from desearch_bot.suffixes import PublicSuffixList, tld_group


def reason(host, categories=None):
    return exclusion_reason(host, categories or {}, tld_group(host))


def test_content_sites_are_kept():
    for host in (
        "nytimes.com",
        "theguardian.com",
        "python.org",
        "stripe.com",
        "autism.org.uk",
    ):
        assert reason(host) is None, host


def test_search_and_social_platforms_are_dropped():
    for host in ("google.com", "youtube.com", "reddit.com", "facebook.com"):
        assert reason(host) == "dynamic_platform", host


def test_infrastructure_is_dropped_by_name_or_list():
    assert reason("gstatic.com") == "infrastructure"
    assert reason("cdn.example.com") == "infrastructure_name"


def test_non_english_country_domains_are_dropped():
    assert reason("spiegel.de") == "non_english_tld"


def test_categories_match_the_exact_host_not_the_parent():
    categories = {"adult": {"blog.news-publisher.com"}}
    assert reason("blog.news-publisher.com", categories) == "ut1_adult"
    assert reason("news-publisher.com", categories) is None


def test_public_suffix_handling(tmp_path):
    rules = tmp_path / "psl.dat"
    rules.write_text("// comment\ncom\nuk\nco.uk\ngithub.io\n", encoding="utf-8")
    psl = PublicSuffixList(rules)
    assert psl.registrable("www.example.com") == "example.com"
    assert psl.registrable("a.b.example.co.uk") == "example.co.uk"
    assert psl.registrable("project.github.io") == "project.github.io"
    assert psl.registrable("com") is None
