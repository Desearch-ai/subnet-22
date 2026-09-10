import random
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path

from desearch_bot import db
from desearch_bot.loop import CRASH_RETRY, adopted, crashed, plan
from desearch_bot.schedule import Trust
from desearch_bot.states import PUBLIC, RETRY, Outcome, State
from desearch_bot.visit import Known, KnownSitemap, SitemapUpdate, Visit

NOW = datetime(2026, 9, 10, 12, 0, tzinfo=timezone.utc)
SCHEMA = Path(db.__file__).with_name("schema.sql").read_text()


def _sitemap(url, next_check_at, sitemap_id=1):
    return KnownSitemap(
        sitemap_id,
        url,
        "urlset",
        0,
        None,
        None,
        None,
        None,
        timedelta(hours=6),
        next_check_at,
        Trust.UNKNOWN,
        None,
    )


def test_a_new_domain_that_answers_is_due_again_when_its_first_sitemap_is():
    update = SitemapUpdate(
        1,
        "https://example.com/s.xml",
        "urlset",
        0,
        None,
        next_check_at=NOW + timedelta(hours=1),
    )
    write = plan(
        Known("example.com"),
        Visit("example.com", sitemaps=[update], requests=3),
        NOW,
        random.Random(0),
    )
    assert write.state is State.ACTIVE
    assert write.next_due_at == NOW + timedelta(hours=1)
    assert write.last_ok_at == NOW


def test_an_active_domain_is_due_at_its_earliest_sitemap_not_its_daily_robots_check():
    known = Known(
        "example.com",
        State.ACTIVE,
        last_ok_at=NOW,
        sitemaps={
            "https://example.com/a.xml": _sitemap(
                "https://example.com/a.xml", NOW + timedelta(days=3)
            ),
            "https://example.com/news.xml": _sitemap(
                "https://example.com/news.xml", NOW + timedelta(minutes=20), 2
            ),
        },
    )
    write = plan(known, Visit("example.com"), NOW, random.Random(0))
    assert write.next_due_at == NOW + timedelta(minutes=20)


def test_sitemaps_left_unread_by_the_file_limit_keep_the_domain_due():
    known = Known(
        "example.com",
        State.ACTIVE,
        last_ok_at=NOW,
        sitemaps={
            "https://example.com/a.xml": _sitemap(
                "https://example.com/a.xml", NOW - timedelta(hours=1)
            ),
        },
    )
    assert plan(known, Visit("example.com"), NOW, random.Random(0)).next_due_at <= NOW


def test_an_active_domain_that_stops_answering_is_retried_on_the_failure_schedule():
    known = Known("example.com", State.ACTIVE, last_ok_at=NOW - timedelta(hours=2))
    write = plan(
        known, Visit("example.com", Outcome.UNREACHABLE, reason="timeout"), NOW
    )
    assert write.state is State.FAILING and write.next_due_at == NOW + RETRY[0]
    assert write.last_ok_at == NOW - timedelta(hours=2)


def test_a_redirect_records_its_destination_and_a_later_normal_answer_clears_it():
    redirected = plan(
        Known("old.com"),
        Visit("old.com", Outcome.REDIRECT, canonical_host="new.com"),
        NOW,
    )
    assert redirected.canonical_host == "new.com"
    back = plan(
        Known("old.com", State.REDIRECTS, canonical_host="new.com"),
        Visit("old.com", Outcome.NO_SITEMAP),
        NOW,
    )
    assert back.canonical_host is None


def test_losing_contact_does_not_forget_where_a_domain_redirects():
    write = plan(
        Known("old.com", State.REDIRECTS, canonical_host="new.com"),
        Visit("old.com", Outcome.UNREACHABLE),
        NOW,
    )
    assert write.canonical_host == "new.com"


def test_an_excluded_domain_is_never_due_again():
    write = plan(Known("example.com"), Visit("example.com", Outcome.EXCLUDED), NOW)
    assert write.state is State.EXCLUDED and write.next_due_at is None


def test_a_visit_that_crashed_on_our_side_changes_nothing_but_the_retry_time():
    known = Known("example.com", State.ACTIVE, failures=0, last_ok_at=NOW)
    write = crashed(known, "KeyError", NOW)
    assert (write.state, write.failures, write.last_ok_at) == (State.ACTIVE, 0, NOW)
    assert write.next_due_at == NOW + CRASH_RETRY
    assert write.reason == "crashed: KeyError"


def test_the_published_view_lists_exactly_the_public_states():
    view = SCHEMA[SCHEMA.index("CREATE OR REPLACE VIEW published_domains") :]
    assert set(re.findall(r"'(\w+)'", view.split("AND")[0])) == {
        s.value for s in PUBLIC
    }


def test_the_state_constraint_allows_every_state_and_nothing_else():
    start = SCHEMA.index("domains_state_check")
    check = SCHEMA[start : SCHEMA.index("))", start)]
    assert set(re.findall(r"'(\w+)'", check)) == {s.value for s in State}


def test_every_state_but_excluded_belongs_to_exactly_one_scheduling_tier():
    tiers = [*db.REFRESH, *db.DISCOVERY]
    assert sorted(tiers) == sorted(s.value for s in State if s is not State.EXCLUDED)


def test_each_scheduling_tier_has_a_partial_index_the_planner_can_match():
    for tier in (db.REFRESH, db.DISCOVERY):
        assert "WHERE state IN (" + ", ".join(f"'{s}'" for s in tier) + ")" in SCHEMA


def test_a_visit_that_made_no_requests_does_not_count_as_contact():
    known = Known("example.com", State.ACTIVE, last_ok_at=NOW - timedelta(hours=5))
    write = plan(known, Visit("example.com"), NOW, random.Random(0))
    assert write.last_ok_at == known.last_ok_at


def test_a_redirect_destination_joins_the_list_unless_a_rule_excludes_it():
    assert adopted("landing.com") == ("big_generic", State.NEW, None)
    _, state, reason = adopted("cdn.example.com")
    assert state is State.EXCLUDED and reason
