from datetime import datetime

import pytz

from desearch.protocol import ResultType, ScraperStreamingSynapse, ScraperTextRole
from neurons.validators.utils.response_checks import (
    check_markdown_structure,
    collect_summary_sources,
    extract_markdown_links,
    first_duplicate_id,
    is_descending_by_created_at,
    normalize_source_url,
    parse_tweet_date,
    tweet_date_in_range,
    verify_summary_links,
)


def test_normalize_source_url_lowercases():
    assert normalize_source_url("HTTPS://X.com/Foo") == "https://x.com/foo"


def test_normalize_source_url_strips_www_https():
    assert (
        normalize_source_url("https://www.reddit.com/r/foo")
        == "https://reddit.com/r/foo"
    )


def test_normalize_source_url_strips_www_http():
    assert normalize_source_url("http://www.r.com/a") == "http://r.com/a"


def test_normalize_source_url_strips_trailing_slash():
    assert normalize_source_url("https://x.com/a/") == "https://x.com/a"


def test_normalize_source_url_keeps_query_string():
    assert (
        normalize_source_url("https://news.com/x?ref=email")
        == "https://news.com/x?ref=email"
    )


def test_normalize_source_url_keeps_scheme_distinction():
    assert normalize_source_url("http://x.com/a") != normalize_source_url(
        "https://x.com/a"
    )


def test_normalize_source_url_handles_empty():
    assert normalize_source_url("") == ""
    assert normalize_source_url(None) == ""


def test_extract_markdown_links_basic():
    text = "See [a](https://example.com/1) and [b](https://example.com/2)"
    assert extract_markdown_links(text) == [
        ("a", "https://example.com/1"),
        ("b", "https://example.com/2"),
    ]


def test_extract_markdown_links_no_links():
    assert extract_markdown_links("plain text") == []
    assert extract_markdown_links("") == []
    assert extract_markdown_links(None) == []


def test_check_markdown_structure_good():
    ok, issues = check_markdown_structure("**Header**\nsome body")
    assert ok is True
    assert issues == []


def test_check_markdown_structure_hash_header_rejected():
    ok, issues = check_markdown_structure("# Bad header\nbody")
    assert ok is False
    assert "Uses # headers instead of **" in issues


def test_check_markdown_structure_missing_bold_header():
    ok, issues = check_markdown_structure("plain text with no header")
    assert ok is False
    assert "No proper headers found (should use ** for headers)" in issues


def test_check_markdown_structure_empty():
    ok, issues = check_markdown_structure("")
    assert ok is False
    assert "Empty response" in issues


def test_parse_tweet_date_twitter_format():
    dt = parse_tweet_date("Mon May 10 12:00:00 +0000 2026")
    assert dt == datetime(2026, 5, 10, 12, 0, 0, tzinfo=pytz.UTC)


def test_parse_tweet_date_iso_format():
    dt = parse_tweet_date("2026-05-10T12:00:00Z")
    assert dt == datetime(2026, 5, 10, 12, 0, 0, tzinfo=pytz.UTC)


def test_parse_tweet_date_iso_with_offset():
    dt = parse_tweet_date("2026-05-10T12:00:00+00:00")
    assert dt == datetime(2026, 5, 10, 12, 0, 0, tzinfo=pytz.UTC)


def test_parse_tweet_date_returns_none_on_garbage():
    assert parse_tweet_date("not a date") is None
    assert parse_tweet_date("") is None
    assert parse_tweet_date(None) is None


def test_tweet_date_in_range_inclusive():
    start = "2026-05-01T00:00:00Z"
    end = "2026-05-31T23:59:59Z"
    assert tweet_date_in_range("Mon May 15 12:00:00 +0000 2026", start, end) is True


def test_tweet_date_in_range_before_start():
    assert (
        tweet_date_in_range(
            "Mon Apr 10 12:00:00 +0000 2026",
            "2026-05-01T00:00:00Z",
            "2026-05-31T23:59:59Z",
        )
        is False
    )


def test_tweet_date_in_range_after_end():
    assert (
        tweet_date_in_range(
            "Mon Jun 10 12:00:00 +0000 2026",
            "2026-05-01T00:00:00Z",
            "2026-05-31T23:59:59Z",
        )
        is False
    )


def test_tweet_date_in_range_missing_start_bound():
    assert (
        tweet_date_in_range(
            "Mon Apr 10 12:00:00 +0000 2026",
            None,
            "2026-05-31T23:59:59Z",
        )
        is True
    )


def test_tweet_date_in_range_missing_end_bound():
    assert (
        tweet_date_in_range(
            "Mon Jun 10 12:00:00 +0000 2026",
            "2026-05-01T00:00:00Z",
            None,
        )
        is True
    )


def test_tweet_date_in_range_missing_both_bounds():
    assert tweet_date_in_range("Mon May 10 12:00:00 +0000 2026", None, None) is True


def test_tweet_date_in_range_unparseable_date_returns_false():
    assert tweet_date_in_range("garbage", "2026-05-01T00:00:00Z", None) is False


def test_first_duplicate_id_no_duplicates():
    assert first_duplicate_id([{"id": "1"}, {"id": "2"}, {"id": "3"}]) is None


def test_first_duplicate_id_finds_first_duplicate():
    assert first_duplicate_id([{"id": "1"}, {"id": "2"}, {"id": "1"}]) == "1"


def test_first_duplicate_id_custom_key():
    items = [{"url": "https://a"}, {"url": "https://b"}, {"url": "https://a"}]
    assert first_duplicate_id(items, key="url") == "https://a"


def test_first_duplicate_id_skips_missing_key():
    assert first_duplicate_id([{"id": None}, {"id": "1"}, {"id": "1"}]) == "1"
    assert first_duplicate_id([{"id": "1"}, {"other": "x"}, {"id": "1"}]) == "1"


def test_first_duplicate_id_empty_list():
    assert first_duplicate_id([]) is None
    assert first_duplicate_id(None) is None


def test_is_descending_by_created_at_sorted():
    items = [
        {"created_at": "Tue May 15 12:00:00 +0000 2026"},
        {"created_at": "Mon May 10 12:00:00 +0000 2026"},
        {"created_at": "Sun May 05 12:00:00 +0000 2026"},
    ]
    assert is_descending_by_created_at(items) is True


def test_is_descending_by_created_at_unsorted():
    items = [
        {"created_at": "Mon May 10 12:00:00 +0000 2026"},
        {"created_at": "Tue May 15 12:00:00 +0000 2026"},
    ]
    assert is_descending_by_created_at(items) is False


def test_is_descending_by_created_at_missing_date_returns_false():
    items = [{"created_at": "Tue May 15 12:00:00 +0000 2026"}, {"id": "no date"}]
    assert is_descending_by_created_at(items) is False


def test_is_descending_by_created_at_garbage_date_returns_false():
    items = [
        {"created_at": "Tue May 15 12:00:00 +0000 2026"},
        {"created_at": "garbage"},
    ]
    assert is_descending_by_created_at(items) is False


def test_is_descending_by_created_at_empty_is_true():
    assert is_descending_by_created_at([]) is True


def _tweet_dict(tid: str, username: str = "foo") -> dict:
    return {"id": tid, "user": {"username": username}}


def test_collect_summary_sources_tweets_and_search_results():
    response = ScraperStreamingSynapse(
        prompt="x",
        miner_tweets=[_tweet_dict("123"), _tweet_dict("456", "bar")],
        search_results=[
            {"title": "T", "link": "https://news.com/article", "snippet": "s"}
        ],
    )
    sources = collect_summary_sources(response)
    assert "https://x.com/foo/status/123" in sources
    assert "https://x.com/bar/status/456" in sources
    assert "https://news.com/article" in sources


def test_collect_summary_sources_normalizes_www():
    response = ScraperStreamingSynapse(
        prompt="x",
        search_results=[
            {"title": "T", "link": "https://www.reddit.com/r/foo", "snippet": "s"}
        ],
    )
    assert "https://reddit.com/r/foo" in collect_summary_sources(response)


def test_collect_summary_sources_skips_non_dict_tweets():
    """Defensive: even if a non-dict slips into miner_tweets, it's skipped."""
    response = ScraperStreamingSynapse.model_construct(
        prompt="x", miner_tweets=[_tweet_dict("good"), "garbage"]
    )
    sources = collect_summary_sources(response)
    assert sources == {"https://x.com/foo/status/good"}


def test_collect_summary_sources_empty():
    response = ScraperStreamingSynapse(prompt="x")
    assert collect_summary_sources(response) == set()


def test_verify_summary_links_all_matched():
    response = ScraperStreamingSynapse(
        prompt="x",
        text_chunks={
            ScraperTextRole.FINAL_SUMMARY.value: [
                "**Header**\n[a](https://x.com/foo/status/123)"
            ]
        },
        miner_tweets=[_tweet_dict("123")],
        result_type=ResultType.LINKS_WITH_FINAL_SUMMARY,
    )
    assert verify_summary_links(response) == (1, 1)


def test_verify_summary_links_partial_match():
    response = ScraperStreamingSynapse(
        prompt="x",
        text_chunks={
            ScraperTextRole.FINAL_SUMMARY.value: [
                "**Header**\n[a](https://x.com/foo/status/123)\n[b](https://nope.com/)"
            ]
        },
        miner_tweets=[_tweet_dict("123")],
        result_type=ResultType.LINKS_WITH_FINAL_SUMMARY,
    )
    assert verify_summary_links(response) == (1, 2)


def test_verify_summary_links_no_links():
    response = ScraperStreamingSynapse(
        prompt="x",
        text_chunks={ScraperTextRole.FINAL_SUMMARY.value: ["**Header**\nplain text"]},
        miner_tweets=[_tweet_dict("123")],
        result_type=ResultType.LINKS_WITH_FINAL_SUMMARY,
    )
    assert verify_summary_links(response) == (0, 0)


def test_verify_summary_links_no_summary():
    response = ScraperStreamingSynapse(prompt="x", miner_tweets=[_tweet_dict("123")])
    assert verify_summary_links(response) == (0, 0)
