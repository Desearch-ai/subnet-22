from datetime import datetime, timedelta, timezone

import pytest

from neurons.validators.scoring import capacity, miner_db
from neurons.validators.scoring.capacity import (
    DECAY_FRACTION,
    DEFAULT_PER_UID,
    HARD_CAP_PER_UID,
    QUALITY_THRESHOLD,
    RAMP_FRACTION,
    UNREACHABLE_DECAY_INTERVAL_SEC,
    UNREACHABLE_FAILURE_THRESHOLD,
    decay_unreachable_tick,
    next_verified,
    note_call_result,
)


def test_ramp_up_when_quality_passes():
    """At quality >= threshold, verified grows by max(1, declared * RAMP_FRACTION)."""
    assert next_verified(current=1, declared=100, quality_avg=0.9) == 11
    assert next_verified(current=50, declared=100, quality_avg=0.5) == 60


def test_ramp_down_when_quality_fails():
    """Below threshold, decay step = DECAY_FRACTION * declared (faster than ramp)."""
    assert next_verified(current=50, declared=100, quality_avg=0.1) == 30
    assert DECAY_FRACTION > RAMP_FRACTION


def test_decay_floors_at_default():
    assert next_verified(current=5, declared=100, quality_avg=0.0) == DEFAULT_PER_UID


def test_ramp_caps_at_declared():
    assert next_verified(current=95, declared=100, quality_avg=0.9) == 100
    assert next_verified(current=24, declared=25, quality_avg=0.9) == 25


def test_ramp_caps_at_hard_cap():
    assert next_verified(current=95, declared=500, quality_avg=0.9) == HARD_CAP_PER_UID


def test_ramp_step_minimum_one_for_small_declared():
    """5 * 0.10 = 0.5, rounds to 0 — but ramp still moves by 1 per epoch."""
    assert int(5 * RAMP_FRACTION) == 0
    assert next_verified(current=1, declared=5, quality_avg=0.9) == 2
    assert next_verified(current=3, declared=5, quality_avg=0.0) == 2


def test_threshold_exactly_qualifies():
    assert next_verified(current=10, declared=100, quality_avg=QUALITY_THRESHOLD) == 20
    assert (
        next_verified(current=10, declared=100, quality_avg=QUALITY_THRESHOLD - 0.001)
        == 0
        or next_verified(
            current=10, declared=100, quality_avg=QUALITY_THRESHOLD - 0.001
        )
        == DEFAULT_PER_UID
    )


def test_declared_below_default_clamps():
    """Bogus declared=0 is treated as DEFAULT_PER_UID for step purposes."""
    assert next_verified(current=1, declared=0, quality_avg=0.9) == DEFAULT_PER_UID


@pytest.fixture
async def db(tmp_path):
    await miner_db.initialize(str(tmp_path / "miner.db"), readonly=False, owner=True)
    yield miner_db
    await miner_db.close()


async def _seed_unreachable(uid: int, verified: int, unreachable_since: str):
    await miner_db.register_miner(
        uid=uid, search_type="x_search", declared=100, hotkey=f"h{uid}", coldkey="c"
    )
    await miner_db.bulk_update_verified("x_search", {uid: verified})
    async with miner_db._conn() as conn:
        await conn.execute(
            "UPDATE miner_concurrency SET unreachable_since=?, last_decay_at=? "
            "WHERE uid=? AND search_type='x_search'",
            (unreachable_since, unreachable_since, uid),
        )
        await conn.commit()


async def test_decay_single_tick_after_one_interval(db):
    """One full interval elapsed → exactly one 10% cut."""
    past = (
        datetime.now(timezone.utc)
        - timedelta(seconds=UNREACHABLE_DECAY_INTERVAL_SEC + 1)
    ).isoformat()
    await _seed_unreachable(uid=1, verified=100, unreachable_since=past)

    await decay_unreachable_tick()

    row = await miner_db.get_concurrency_row(1, "x_search")
    assert row["verified"] == 90


async def test_decay_compounds_over_multiple_intervals(db):
    """3 intervals elapsed → 100 -> 90 -> 81 -> 72 (int truncation)."""
    past = (
        datetime.now(timezone.utc)
        - timedelta(seconds=3 * UNREACHABLE_DECAY_INTERVAL_SEC + 1)
    ).isoformat()
    await _seed_unreachable(uid=2, verified=100, unreachable_since=past)

    await decay_unreachable_tick()

    row = await miner_db.get_concurrency_row(2, "x_search")
    assert row["verified"] == 72


async def test_decay_noop_before_first_interval(db):
    """Less than one interval elapsed → no change."""
    past = (
        datetime.now(timezone.utc)
        - timedelta(seconds=UNREACHABLE_DECAY_INTERVAL_SEC - 30)
    ).isoformat()
    await _seed_unreachable(uid=3, verified=50, unreachable_since=past)

    await decay_unreachable_tick()

    row = await miner_db.get_concurrency_row(3, "x_search")
    assert row["verified"] == 50


async def test_decay_floors_at_one(db):
    """Long enough outage drives verified to 1, not below."""
    past = (
        datetime.now(timezone.utc)
        - timedelta(seconds=200 * UNREACHABLE_DECAY_INTERVAL_SEC)
    ).isoformat()
    await _seed_unreachable(uid=4, verified=10, unreachable_since=past)

    await decay_unreachable_tick()

    row = await miner_db.get_concurrency_row(4, "x_search")
    assert row["verified"] == 1


async def test_decay_skips_reachable_miners(db):
    """A reachable miner (no unreachable_since) is left alone."""
    await miner_db.register_miner(
        uid=5, search_type="x_search", declared=100, hotkey="h5", coldkey="c"
    )
    await miner_db.bulk_update_verified("x_search", {5: 50})

    await decay_unreachable_tick()

    row = await miner_db.get_concurrency_row(5, "x_search")
    assert row["verified"] == 50


async def test_recovery_clears_last_decay_at(db):
    """Recovery clears unreachable_since + last_decay_at, preserves decayed verified."""
    past = (
        datetime.now(timezone.utc)
        - timedelta(seconds=UNREACHABLE_DECAY_INTERVAL_SEC + 1)
    ).isoformat()
    await _seed_unreachable(uid=6, verified=80, unreachable_since=past)
    await decay_unreachable_tick()

    await note_call_result(uid=6, search_type="x_search", success=True)

    row = await miner_db.get_concurrency_row(6, "x_search")
    assert row["unreachable_since"] is None
    assert row["last_decay_at"] is None
    assert row["verified"] == 72  # decayed value preserved across recovery
