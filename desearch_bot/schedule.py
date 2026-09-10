"""How often to re-read a sitemap, and whether to believe its dates."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from datetime import datetime, timedelta, timezone
from enum import StrEnum

MIN_INTERVAL = timedelta(minutes=10)
MAX_INTERVAL = timedelta(days=7)
DEFAULT_INTERVAL = timedelta(days=1)
NEWS_INTERVAL = timedelta(hours=1)

DECLARED = {
    "always": MIN_INTERVAL,
    "hourly": timedelta(hours=1),
    "daily": timedelta(days=1),
    "weekly": MAX_INTERVAL,
    "monthly": MAX_INTERVAL,
    "yearly": MAX_INTERVAL,
    "never": MAX_INTERVAL,
}

# A change pulls the interval in harder than a quiet read pushes it out.
FASTER = 0.5
SLOWER = 1.5

MIN_DATED = 20
AGREEMENT = 0.9
STAMP_WINDOW = timedelta(hours=1)
EARLIEST = datetime(1995, 1, 1, tzinfo=timezone.utc)
FUTURE_SLACK = timedelta(days=1)


class Trust(StrEnum):
    UNKNOWN = "unknown"
    SUSPECT = "suspect"
    TRUSTED = "trusted"
    UNTRUSTED = "untrusted"


def first_interval(changefreq: str | None, news: bool = False) -> timedelta:
    """Where a new sitemap's schedule starts, before its behaviour is known."""
    declared = DECLARED.get((changefreq or "").strip().lower(), DEFAULT_INTERVAL)
    return min(declared, NEWS_INTERVAL) if news else declared


def next_interval(current: timedelta, changed: bool) -> timedelta:
    """Look sooner at a file that changed, and later at one that did not."""
    scaled = current * (FASTER if changed else SLOWER)
    return max(MIN_INTERVAL, min(MAX_INTERVAL, scaled))


def plausible(date: datetime | None, fetched_at: datetime) -> datetime | None:
    if date is None or date < EARLIEST or date > fetched_at + FUTURE_SLACK:
        return None
    return date


def assess_dates(
    dates: Iterable[datetime | None], fetched_at: datetime, previous: Trust
) -> Trust:
    """Whether a sitemap's lastmod values carry information, judged from one read."""
    dated = [d for d in (plausible(d, fetched_at) for d in dates) if d is not None]
    if len(dated) < MIN_DATED:
        return previous
    if Counter(dated).most_common(1)[0][1] >= AGREEMENT * len(dated):
        return Trust.UNTRUSTED
    stamped = sum(1 for d in dated if abs(fetched_at - d) <= STAMP_WINDOW)
    if stamped >= AGREEMENT * len(dated):
        return Trust.UNTRUSTED if previous is Trust.SUSPECT else Trust.SUSPECT
    return Trust.TRUSTED


def relies_on_dates(trust: Trust) -> bool:
    return trust is Trust.TRUSTED
