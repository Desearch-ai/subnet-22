from datetime import datetime, timedelta, timezone

from desearch_bot.schedule import (
    DEFAULT_INTERVAL,
    MAX_INTERVAL,
    MIN_DATED,
    MIN_INTERVAL,
    NEWS_INTERVAL,
    Trust,
    assess_dates,
    first_interval,
    next_interval,
    relies_on_dates,
)

FETCHED = datetime(2026, 9, 10, 12, 0, tzinfo=timezone.utc)


def _varied(count=40):
    return [FETCHED - timedelta(days=i, hours=i % 7) for i in range(1, count + 1)]


def test_the_declared_frequency_seeds_the_first_interval():
    assert first_interval("hourly") == timedelta(hours=1)
    assert first_interval("daily") == timedelta(days=1)
    assert first_interval("weekly") == MAX_INTERVAL
    assert first_interval("always") == MIN_INTERVAL
    assert first_interval(None) == DEFAULT_INTERVAL
    assert first_interval("sometimes") == DEFAULT_INTERVAL


def test_news_starts_fast_but_never_slower_than_the_site_declares():
    assert first_interval("daily", news=True) == NEWS_INTERVAL
    assert first_interval("always", news=True) == MIN_INTERVAL


def test_a_file_that_keeps_changing_is_read_more_often_down_to_the_floor():
    interval = timedelta(days=1)
    for _ in range(20):
        interval = next_interval(interval, changed=True)
    assert interval == MIN_INTERVAL


def test_a_file_that_never_changes_drifts_out_to_the_ceiling():
    interval = timedelta(hours=1)
    for _ in range(20):
        interval = next_interval(interval, changed=False)
    assert interval == MAX_INTERVAL


def test_one_change_outweighs_one_quiet_read():
    base = timedelta(hours=8)
    assert next_interval(next_interval(base, changed=True), changed=False) < base


def test_dates_that_vary_are_trusted():
    trust = assess_dates(_varied(), FETCHED, Trust.UNKNOWN)
    assert trust is Trust.TRUSTED and relies_on_dates(trust)


def test_the_same_date_on_every_page_is_not_trusted():
    assert (
        assess_dates([FETCHED - timedelta(days=3)] * 40, FETCHED, Trust.UNKNOWN)
        is Trust.UNTRUSTED
    )


def test_stamping_the_fetch_time_is_suspect_once_and_untrusted_twice():
    stamped = [FETCHED - timedelta(seconds=i) for i in range(40)]
    first = assess_dates(stamped, FETCHED, Trust.UNKNOWN)
    assert first is Trust.SUSPECT and not relies_on_dates(first)
    assert assess_dates(stamped, FETCHED, first) is Trust.UNTRUSTED


def test_a_suspect_sitemap_with_real_dates_next_time_is_trusted():
    assert assess_dates(_varied(), FETCHED, Trust.SUSPECT) is Trust.TRUSTED


def test_too_few_dates_keeps_the_previous_verdict():
    few = _varied(MIN_DATED - 1)
    assert assess_dates(few, FETCHED, Trust.TRUSTED) is Trust.TRUSTED
    assert assess_dates(few, FETCHED, Trust.UNKNOWN) is Trust.UNKNOWN


def test_impossible_dates_are_ignored_rather_than_believed():
    junk = [datetime(1970, 1, 1, tzinfo=timezone.utc)] * 30 + [
        FETCHED + timedelta(days=400)
    ] * 30
    assert assess_dates(junk, FETCHED, Trust.UNKNOWN) is Trust.UNKNOWN
