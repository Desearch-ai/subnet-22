import tempfile
from pathlib import Path

import pytest

from app.budget import COVERAGE_GATE, Budgets


@pytest.fixture
def budgets():
    with tempfile.TemporaryDirectory() as d:
        yield Budgets(str(Path(d) / "b.db"))


def work(budgets, hotkey, assigned, returned, verified):
    budgets.assign(hotkey, assigned)
    budgets.returned(hotkey, returned)
    if verified:
        budgets.reward(hotkey, "t", verified)


def test_full_coverage_earns_a_share(budgets):
    work(budgets, "a", 100, 100, 100)
    assert budgets.shares() == {"a": 1.0}


def test_below_the_gate_earns_nothing(budgets):
    work(budgets, "a", 100, 100, 100)
    work(budgets, "omitter", 100, 80, 80)
    shares = budgets.shares()
    assert "omitter" not in shares
    assert shares == {"a": 1.0}


def test_just_above_the_gate_still_earns(budgets):
    work(budgets, "a", 100, 100, 100)
    work(budgets, "b", 100, 86, 86)
    assert set(budgets.shares()) == {"a", "b"}


def test_shares_are_proportional_to_verified_work(budgets):
    work(budgets, "big", 1000, 1000, 1000)
    work(budgets, "small", 100, 100, 100)
    shares = budgets.shares()
    assert shares["big"] == pytest.approx(10 * shares["small"])


def test_a_miner_that_returns_nothing_is_excluded(budgets):
    work(budgets, "a", 100, 100, 100)
    budgets.assign("hoarder", 500)
    assert "hoarder" not in budgets.shares()
    assert budgets.coverage_report()["hoarder"]["coverage"] == 0.0


def test_gate_is_on_completeness_not_correctness(budgets):
    """A miner returning every URL with fabricated bodies still passes coverage.

    Correctness is the sampled re-fetch's job; conflating them would let an omitter hide.
    """
    work(budgets, "fabricator", 100, 100, 100)
    assert budgets.coverage_report()["fabricator"]["eligible"]
    assert COVERAGE_GATE == 0.85
