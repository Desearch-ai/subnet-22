from app.budget import (
    HOUR,
    LOCKOUT_STEPS_H,
    STRIKE_BURST_S,
    STRIKE_WINDOW_H,
    Budgets,
)
from app.state import connect

NOW = 1_800_000_000.0
FIRST, SECOND, THIRD = (hours * HOUR for hours in LOCKOUT_STEPS_H)


def budgets(tmp_path) -> Budgets:
    return Budgets(connect(str(tmp_path / "b.db")))


def test_one_strike_is_a_warning_and_the_second_locks_the_miner_out(tmp_path):
    store = budgets(tmp_path)

    assert store.strike("m", "content_mismatch", "t1", judged=1, now=NOW) is None
    assert store.locked_until("m", now=NOW) is None

    until = store.strike("m", "errors_not_reproducible", "t2", judged=2, now=NOW + HOUR)
    assert until == NOW + HOUR + FIRST
    assert store.locked_until("m", now=NOW + HOUR + 60) == until
    assert store.locked_until("m", now=until) is None
    assert store.locked_until("other", now=NOW + HOUR + 60) is None


def test_a_busy_miner_with_a_few_bad_batches_is_not_locked(tmp_path):
    store = budgets(tmp_path)
    store.strike("m", "content_mismatch", "t1", judged=50, now=NOW)

    assert store.strike("m", "content_mismatch", "t2", judged=100, now=NOW + 60) is None
    assert store.strike("m", "coverage", "t3", judged=60, now=NOW + 120) is not None


def test_strikes_a_day_apart_never_lock(tmp_path):
    store = budgets(tmp_path)
    store.strike("m", "content_mismatch", "t1", judged=1, now=NOW)

    later = NOW + STRIKE_WINDOW_H * HOUR + 1
    assert store.strike("m", "content_mismatch", "t2", judged=1, now=later) is None


def test_each_lockout_within_a_week_lasts_longer(tmp_path):
    store = budgets(tmp_path)
    lengths, now = [], NOW
    for step in range(4):
        store.strike("m", "coverage", f"a{step}", judged=0, now=now)
        until = store.strike("m", "coverage", f"b{step}", judged=0, now=now + 60)
        lengths.append(until - (now + 60))
        now = until + STRIKE_WINDOW_H * HOUR

    assert lengths == [FIRST, SECOND, THIRD, THIRD]


def test_claims_that_lapse_together_are_one_strike(tmp_path):
    store = budgets(tmp_path)
    for n in range(5):
        assert (
            store.strike("m", "claim_expired", f"t{n}", judged=10, now=NOW + n) is None
        )

    later = NOW + STRIKE_BURST_S + 1
    assert store.strike("m", "abandoned", "t9", judged=10, now=later) == later + FIRST


def test_a_hotkey_that_only_hoards_is_locked_out_on_its_second_lapse(tmp_path):
    store = budgets(tmp_path)
    store.strike("h", "claim_expired", "t1", judged=0, now=NOW)

    until = store.strike("h", "claim_expired", "t2", judged=0, now=NOW + 600)
    assert until == NOW + 600 + FIRST


def test_failed_uploads_are_never_merged_into_one_strike(tmp_path):
    store = budgets(tmp_path)
    store.strike("m", "content_mismatch", "t1", judged=2, now=NOW)

    assert store.strike("m", "content_mismatch", "t2", judged=2, now=NOW + 1)
