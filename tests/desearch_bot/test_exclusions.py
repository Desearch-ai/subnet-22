from desearch_bot.exclusions import blocked_operator, exclusion_reason, valid_host
from desearch_bot.suffixes import PublicSuffixList, tld_group


def reason(host, categories=None, adult=frozenset()):
    return exclusion_reason(host, categories or {}, tld_group(host), adult)


def test_content_sites_are_kept():
    for host in (
        "nytimes.com",
        "theguardian.com",
        "python.org",
        "stripe.com",
        "autism.org.uk",
    ):
        assert reason(host) is None, host


def test_adult_domains_come_from_the_blocklists():
    adult = {"somethingexplicit.com"}
    assert reason("somethingexplicit.com", adult=adult) == "adult_list"
    assert reason("analyticsvidhya.com", adult=adult) is None
    assert reason("sussex.ac.uk", adult=adult) is None


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


def test_a_blocked_operator_covers_its_whole_family():
    for host in ("shinhan.com", "shinhanbank.com", "shinhan.ca", "ezshinhancard.com",
                 "newshinhancard.com", "shinhanfinancialgroup.com"):
        assert reason(host) == "blocked_operator", host


def test_unrelated_names_that_contain_the_string_are_not_blocked():
    for host in ("kakushinhan.org", "marushinhanten.com", "tenshinhanten.com"):
        assert blocked_operator(host) is False, host


def test_a_block_wins_over_every_other_rule():
    assert reason("shinhan.ca", {"press": {"shinhan.ca"}}) == "blocked_operator"


def test_filenames_that_parse_as_domains_are_rejected():
    """.sh and .app are real TLDs, so source lists smuggle in shell scripts."""
    for host in ("0_linux.sh", "01_nucdetective_profiler.sh", "2_beta.app", "_wildcard_.ph"):
        assert exclusion_reason(host, {}, "new_generic") == "invalid_host"


def test_real_domains_still_pass_the_hostname_check():
    for host in ("0-0-0checkmate.com", "news-publisher.com", "xn--80ak6aa92e.com", "a.co"):
        assert valid_host(host)
