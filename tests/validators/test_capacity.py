from datetime import datetime, timedelta, timezone

import pytest

from neurons.validators.scoring import capacity, miner_db
from neurons.validators.scoring.capacity import (
    DEFAULT_PER_UID,
    HARD_CAP_PER_UID,
    QUALITY_THRESHOLDS,
    RAMP_FRACTION,
    UNREACHABLE_DECAY_INTERVAL_SEC,
    decay_unreachable_tick,
    next_verified,
    note_call_result,
    passes_combined_gate,
    ramp_after_epoch,
    record_window_quality,
)


def test_ramp_up_when_gate_passes():
    """all_pass=True grows verified by max(1, declared * RAMP_FRACTION)."""
    assert next_verified(current=1, declared=100, all_pass=True) == 11
    assert next_verified(current=50, declared=100, all_pass=True) == 60


def test_ramp_down_when_gate_fails():
    """all_pass=False shrinks verified by DECAY_FRACTION * declared."""
    assert next_verified(current=50, declared=100, all_pass=False) == 40


def test_decay_floors_at_default():
    assert next_verified(current=5, declared=100, all_pass=False) == DEFAULT_PER_UID


def test_ramp_caps_at_declared():
    assert next_verified(current=95, declared=100, all_pass=True) == 100
    assert next_verified(current=24, declared=25, all_pass=True) == 25


def test_ramp_caps_at_hard_cap():
    assert next_verified(current=95, declared=500, all_pass=True) == HARD_CAP_PER_UID


def test_ramp_step_minimum_one_for_small_declared():
    """5 * 0.10 = 0.5 → rounds to 0, but step is floored to 1."""
    assert int(5 * RAMP_FRACTION) == 0
    assert next_verified(current=1, declared=5, all_pass=True) == 2
    assert next_verified(current=3, declared=5, all_pass=False) == 2


def test_declared_below_default_clamps():
    """declared=0 still floors step to 1."""
    assert next_verified(current=1, declared=0, all_pass=True) == DEFAULT_PER_UID


def test_gate_passes_when_all_above_thresholds():
    ema = {
        search_type: threshold + 0.01
        for search_type, threshold in QUALITY_THRESHOLDS.items()
    }
    declared = {"ai_search": 100, "x_search": 100, "web_search": 100}
    assert passes_combined_gate(ema, declared) is True


def test_gate_fails_when_ai_below_threshold():
    ema = {"ai_search": 0.20, "x_search": 0.80, "web_search": 0.80}
    declared = {"ai_search": 100, "x_search": 100, "web_search": 100}
    assert passes_combined_gate(ema, declared) is False


def test_gate_fails_when_x_below_threshold():
    ema = {"ai_search": 0.90, "x_search": 0.40, "web_search": 0.90}
    declared = {"ai_search": 100, "x_search": 100, "web_search": 100}
    assert passes_combined_gate(ema, declared) is False


def test_gate_fails_when_web_below_threshold():
    ema = {"ai_search": 0.90, "x_search": 0.90, "web_search": 0.49}
    declared = {"ai_search": 100, "x_search": 100, "web_search": 100}
    assert passes_combined_gate(ema, declared) is False


def test_gate_fails_when_any_declared_zero():
    """A miner who doesn't advertise AI can never ramp anything."""
    ema = {"ai_search": 0.90, "x_search": 0.90, "web_search": 0.90}
    declared = {"ai_search": 0, "x_search": 100, "web_search": 100}
    assert passes_combined_gate(ema, declared) is False


def test_gate_fails_when_type_missing():
    """If a type is absent from the EMA/declared dicts, treat as fail."""
    ema = {"ai_search": 0.90, "x_search": 0.90}
    declared = {"ai_search": 100, "x_search": 100}
    assert passes_combined_gate(ema, declared) is False


def test_gate_threshold_exact_boundary():
    """quality_avg == threshold qualifies (>= comparison)."""
    ema = {
        "ai_search": QUALITY_THRESHOLDS["ai_search"],
        "x_search": QUALITY_THRESHOLDS["x_search"],
        "web_search": QUALITY_THRESHOLDS["web_search"],
    }
    declared = {"ai_search": 100, "x_search": 100, "web_search": 100}
    assert passes_combined_gate(ema, declared) is True


@pytest.fixture
async def db(tmp_path):
    await miner_db.initialize(str(tmp_path / "miner.db"), readonly=False, owner=True)
    yield miner_db
    await miner_db.close()


async def _register_all_types(uid: int, declared: dict[str, int], hotkey: str = "h"):
    for t, d in declared.items():
        await miner_db.register_miner(
            uid=uid, search_type=t, declared=d, hotkey=hotkey, coldkey="c"
        )


async def _set_ema_and_verified(uid: int, by_type: dict[str, tuple[float, int]]):
    for t, (ema, verified) in by_type.items():
        await miner_db.upsert_quality_avg(uid=uid, search_type=t, quality_avg=ema)
        await miner_db.bulk_update_verified(t, {uid: verified})


async def test_ramp_after_epoch_ramps_all_three_when_gate_passes(db):
    """All EMAs above threshold → all three verifieds step up together."""
    await _register_all_types(
        uid=1, declared={"ai_search": 100, "x_search": 100, "web_search": 100}
    )
    await _set_ema_and_verified(
        uid=1,
        by_type={
            "ai_search": (0.7, 50),
            "x_search": (0.7, 50),
            "web_search": (0.7, 50),
        },
    )

    await ramp_after_epoch([1])

    for t in ("ai_search", "x_search", "web_search"):
        row = await miner_db.get_concurrency_row(1, t)
        assert row["verified"] == 60


async def test_ramp_after_epoch_decays_all_three_when_ai_fails(db):
    """AI EMA below threshold → all three verifieds decay together, even
    though X+Web are individually fine."""
    await _register_all_types(
        uid=2, declared={"ai_search": 100, "x_search": 100, "web_search": 100}
    )
    await _set_ema_and_verified(
        uid=2,
        by_type={
            "ai_search": (0.10, 100),
            "x_search": (0.90, 100),
            "web_search": (0.90, 100),
        },
    )

    await ramp_after_epoch([2])

    for t in ("ai_search", "x_search", "web_search"):
        row = await miner_db.get_concurrency_row(2, t)
        assert row["verified"] == 90


async def test_ramp_after_epoch_decays_when_undeclared_type(db):
    """declared=0 on AI → gate fails → all three decay."""
    await _register_all_types(
        uid=3, declared={"ai_search": 0, "x_search": 100, "web_search": 100}
    )
    await _set_ema_and_verified(
        uid=3,
        by_type={
            "ai_search": (0.0, 1),
            "x_search": (0.90, 100),
            "web_search": (0.90, 100),
        },
    )

    await ramp_after_epoch([3])

    for t in ("x_search", "web_search"):
        row = await miner_db.get_concurrency_row(3, t)
        assert row["verified"] == 90


async def test_ramp_after_epoch_handles_empty_uid_list(db):
    await ramp_after_epoch([])


async def test_record_window_quality_updates_ema_without_ramping(db):
    await _register_all_types(
        uid=4, declared={"ai_search": 100, "x_search": 100, "web_search": 100}
    )
    await _set_ema_and_verified(
        uid=4,
        by_type={
            "ai_search": (0.0, 50),
            "x_search": (0.0, 50),
            "web_search": (0.0, 50),
        },
    )

    await record_window_quality(
        uid=4,
        search_type="ai_search",
        quality=1.0,
        window_start="2026-05-20T00:00:00+00:00",
        allocated=50,
    )

    row = await miner_db.get_concurrency_row(4, "ai_search")
    assert row["quality_avg"] == pytest.approx(capacity.QUALITY_EMA_ALPHA)
    assert row["verified"] == 50


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
    assert row["verified"] == 72


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
