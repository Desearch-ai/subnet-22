from app.budget import HOUR, LOCKOUT_H, STRIKE_WINDOW_H, Budgets
from app.state import connect

NOW = 1_800_000_000.0


def budgets(tmp_path) -> Budgets:
    return Budgets(connect(str(tmp_path / "b.db")))


def test_one_strike_is_a_warning_and_the_second_locks_the_miner_out(tmp_path):
    store = budgets(tmp_path)

    assert store.strike("m", "content_mismatch", "t1", judged=1, now=NOW) is None
    assert store.locked_until("m", now=NOW) is None

    until = store.strike("m", "errors_not_reproducible", "t2", judged=2, now=NOW + HOUR)
    assert until == NOW + HOUR + LOCKOUT_H * HOUR
    assert store.locked_until("m", now=NOW + 2 * HOUR) == until
    assert store.locked_until("m", now=until) is None
    assert store.locked_until("other", now=NOW + 2 * HOUR) is None


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


def test_a_strike_right_after_a_lockout_locks_again(tmp_path):
    store = budgets(tmp_path)
    store.strike("m", "coverage", "t1", judged=1, now=NOW)
    first = store.strike("m", "coverage", "t2", judged=2, now=NOW + 60)

    again = store.strike("m", "coverage", "t3", judged=3, now=first + 60)
    assert again == first + 60 + LOCKOUT_H * HOUR
