import random
from datetime import datetime, timedelta, timezone

from desearch_bot.states import (
    DOWN_AFTER,
    JITTER,
    PUBLIC,
    RECHECK,
    RETRY,
    Outcome,
    State,
    decide,
    serves_miners,
)

NOW = datetime(2026, 9, 10, 12, 0, tzinfo=timezone.utc)


def _within(when, interval, start=NOW):
    return interval * (1 - JITTER) <= when - start <= interval * (1 + JITTER)


def test_a_new_domain_with_a_sitemap_becomes_active():
    d = decide(State.NEW, Outcome.SITEMAP, 0, None, NOW, random.Random(0))
    assert d.state is State.ACTIVE and d.failures == 0
    assert _within(d.next_check_at, RECHECK[State.ACTIVE])


def test_an_active_domain_that_stops_answering_retries_on_a_widening_schedule():
    state, failures, now, waits = State.ACTIVE, 0, NOW, []
    for _ in range(7):
        d = decide(state, Outcome.UNREACHABLE, failures, NOW, now)
        waits.append(d.next_check_at - now)
        state, failures, now = d.state, d.failures, d.next_check_at
    assert waits == [*RETRY, RETRY[-1], RETRY[-1]]
    assert state is State.FAILING


def test_down_for_a_day_then_back_is_active_again_within_one_retry():
    state, failures, now = State.ACTIVE, 0, NOW
    back_at = NOW + timedelta(hours=24)
    while now < back_at:
        d = decide(state, Outcome.UNREACHABLE, failures, NOW, now)
        assert d.state is State.FAILING
        state, failures, now = d.state, d.failures, d.next_check_at
    recovered = decide(state, Outcome.SITEMAP, failures, NOW, now, random.Random(0))
    assert recovered.state is State.ACTIVE and recovered.failures == 0
    assert now - back_at <= RETRY[-1]


def test_failing_for_a_week_goes_down_and_is_checked_daily():
    last_ok = NOW - DOWN_AFTER - timedelta(hours=1)
    d = decide(State.FAILING, Outcome.UNREACHABLE, 30, last_ok, NOW, random.Random(0))
    assert d.state is State.DOWN
    assert _within(d.next_check_at, RECHECK[State.DOWN])


def test_a_down_domain_that_answers_is_active_again():
    d = decide(
        State.DOWN, Outcome.SITEMAP, 60, NOW - timedelta(days=20), NOW, random.Random(0)
    )
    assert d.state is State.ACTIVE and d.failures == 0


def test_a_domain_that_has_never_answered_is_unreachable_and_checked_monthly():
    for previous in (State.NEW, State.UNREACHABLE, State.NO_SITEMAP, State.BLOCKED):
        d = decide(previous, Outcome.UNREACHABLE, 0, None, NOW, random.Random(0))
        assert d.state is State.UNREACHABLE
        assert _within(d.next_check_at, RECHECK[State.UNREACHABLE])


def test_a_domain_that_adds_a_sitemap_later_becomes_active():
    first = decide(State.NEW, Outcome.NO_SITEMAP, 0, None, NOW, random.Random(0))
    assert first.state is State.NO_SITEMAP
    assert _within(first.next_check_at, RECHECK[State.NO_SITEMAP])
    later = decide(State.NO_SITEMAP, Outcome.SITEMAP, 0, NOW, first.next_check_at)
    assert later.state is State.ACTIVE


def test_each_settled_state_waits_its_own_interval():
    for outcome, state in (
        (Outcome.REDIRECT, State.REDIRECTS),
        (Outcome.BLOCKED, State.BLOCKED),
        (Outcome.INELIGIBLE, State.INELIGIBLE),
    ):
        d = decide(State.NEW, outcome, 0, None, NOW, random.Random(0))
        assert d.state is state
        assert _within(d.next_check_at, RECHECK[state])


def test_an_excluded_domain_is_never_checked_again():
    d = decide(State.ACTIVE, Outcome.EXCLUDED, 0, NOW, NOW)
    assert d.state is State.EXCLUDED and d.next_check_at is None


def test_only_active_domains_feed_miners():
    assert [state for state in State if serves_miners(state)] == [State.ACTIVE]


def test_the_public_list_holds_no_state_known_to_be_bad():
    bad = {
        State.DOWN,
        State.UNREACHABLE,
        State.REDIRECTS,
        State.BLOCKED,
        State.INELIGIBLE,
        State.EXCLUDED,
    }
    assert not PUBLIC & bad


def test_jitter_spreads_domains_that_were_visited_together():
    due = {
        decide(
            State.NEW, Outcome.NO_SITEMAP, 0, None, NOW, random.Random(seed)
        ).next_check_at
        for seed in range(50)
    }
    assert len(due) > 40
