"""What a domain is, and when to look at it again."""

from __future__ import annotations

import random
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import StrEnum


class State(StrEnum):
    NEW = "new"
    ACTIVE = "active"
    FAILING = "failing"
    DOWN = "down"
    UNREACHABLE = "unreachable"
    NO_SITEMAP = "no_sitemap"
    REDIRECTS = "redirects"
    BLOCKED = "blocked"
    INELIGIBLE = "ineligible"
    EXCLUDED = "excluded"


class Outcome(StrEnum):
    SITEMAP = "sitemap"
    NO_SITEMAP = "no_sitemap"
    REDIRECT = "redirect"
    BLOCKED = "blocked"
    INELIGIBLE = "ineligible"
    UNREACHABLE = "unreachable"
    EXCLUDED = "excluded"


RECHECK = {
    State.ACTIVE: timedelta(days=1),
    State.DOWN: timedelta(days=1),
    State.UNREACHABLE: timedelta(days=30),
    State.NO_SITEMAP: timedelta(days=30),
    State.REDIRECTS: timedelta(days=30),
    State.BLOCKED: timedelta(days=30),
    State.INELIGIBLE: timedelta(days=90),
}
RETRY = (
    timedelta(minutes=15),
    timedelta(minutes=30),
    timedelta(hours=1),
    timedelta(hours=2),
    timedelta(hours=4),
)
DOWN_AFTER = timedelta(days=7)
# Spread rechecks so domains visited together do not all fall due in the same second.
JITTER = 0.1

SETTLED = {
    Outcome.SITEMAP: State.ACTIVE,
    Outcome.NO_SITEMAP: State.NO_SITEMAP,
    Outcome.REDIRECT: State.REDIRECTS,
    Outcome.BLOCKED: State.BLOCKED,
    Outcome.INELIGIBLE: State.INELIGIBLE,
}

PUBLIC = frozenset({State.NEW, State.ACTIVE, State.FAILING, State.NO_SITEMAP})


@dataclass(frozen=True)
class Decision:
    state: State
    failures: int
    next_check_at: datetime | None


def decide(
    previous: State,
    outcome: Outcome,
    failures: int,
    last_ok_at: datetime | None,
    now: datetime,
    rng: random.Random | None = None,
) -> Decision:
    """The state a visit leaves a domain in, and when to look at it again."""
    if outcome is Outcome.EXCLUDED:
        return Decision(State.EXCLUDED, 0, None)
    if outcome is Outcome.UNREACHABLE:
        return _failed(previous, failures + 1, last_ok_at, now, rng)
    state = SETTLED[outcome]
    return Decision(state, 0, _after(now, RECHECK[state], rng))


def serves_miners(state: State) -> bool:
    return state is State.ACTIVE


def _failed(
    previous: State,
    failures: int,
    last_ok_at: datetime | None,
    now: datetime,
    rng: random.Random | None,
) -> Decision:
    if previous in (State.ACTIVE, State.FAILING):
        if last_ok_at is not None and now - last_ok_at >= DOWN_AFTER:
            return Decision(State.DOWN, failures, _after(now, RECHECK[State.DOWN], rng))
        return Decision(
            State.FAILING, failures, now + RETRY[min(failures, len(RETRY)) - 1]
        )
    if previous is State.DOWN:
        return Decision(State.DOWN, failures, _after(now, RECHECK[State.DOWN], rng))
    return Decision(
        State.UNREACHABLE, failures, _after(now, RECHECK[State.UNREACHABLE], rng)
    )


def _after(now: datetime, interval: timedelta, rng: random.Random | None) -> datetime:
    return now + interval * (rng or random).uniform(1 - JITTER, 1 + JITTER)
