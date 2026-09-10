from datetime import datetime, timedelta, timezone

from desearch_bot.loop import DomainWrite
from desearch_bot.records import (
    known,
    new_domain,
    unread_sitemap,
    url_count,
    written_domain,
    written_sitemap,
)
from desearch_bot.schedule import Trust
from desearch_bot.states import State
from desearch_bot.visit import SitemapUpdate, Visit

NOW = datetime(2026, 9, 10, 12, tzinfo=timezone.utc)
URL = "https://a.com/sitemap.xml"


def _update(changed=True, url_count=12, status="ok"):
    return SitemapUpdate(
        7,
        URL,
        "urlset",
        0,
        None,
        status=status,
        etag='"v1"',
        content_hash="h",
        changed=changed,
        url_count=url_count,
        interval=timedelta(hours=6),
        next_check_at=NOW + timedelta(hours=6),
        trust=Trust.TRUSTED,
    )


def test_a_new_domain_becomes_a_known_domain_with_nothing_learned_yet():
    record = new_domain(40, "big_generic", ["news"], due=NOW)
    state = known("a.com", record, [])
    assert (state.state, state.categories, state.sitemaps) == (
        State.NEW,
        frozenset({"news"}),
        {},
    )


def test_a_written_sitemap_reads_back_as_the_same_known_sitemap():
    record = written_sitemap(None, _update(), NOW)
    [(url, stored)] = known(
        "a.com", new_domain(1, None), [(URL, record)]
    ).sitemaps.items()
    assert (url, stored.id, stored.etag, stored.trust, stored.url_count) == (
        URL,
        7,
        '"v1"',
        Trust.TRUSTED,
        12,
    )
    assert (stored.interval, stored.next_check_at) == (
        timedelta(hours=6),
        NOW + timedelta(hours=6),
    )


def test_an_unchanged_read_keeps_the_last_count():
    first = written_sitemap(None, _update(), NOW)
    later = written_sitemap(
        first, _update(changed=False, url_count=0), NOW + timedelta(hours=6)
    )
    assert later["urls"] == 12 and later["changed"] == first["changed"]


def test_robots_fields_change_only_when_robots_was_read():
    base = new_domain(1, None)
    read = DomainWrite(
        "a.com",
        State.ACTIVE,
        None,
        0,
        NOW,
        NOW,
        None,
        NOW,
        Visit("a.com", robots_read=True, crawl_delay=2.0, language="en"),
    )
    record = written_domain(base, read, 12)
    unread = DomainWrite(
        "a.com", State.ACTIVE, None, 0, NOW, NOW, None, NOW, Visit("a.com")
    )
    later = written_domain(record, unread, 12)
    assert (later["delay"], later["lang"], later["urls"]) == (2.0, "en", 12)


def test_only_sitemaps_that_answered_count_towards_the_domain():
    ok = written_sitemap(None, _update(url_count=12), NOW)
    broken = written_sitemap(None, _update(url_count=30, status="error"), NOW)
    assert url_count([ok, broken, unread_sitemap("https://a.com/s2.xml", 1, 7)]) == 12
