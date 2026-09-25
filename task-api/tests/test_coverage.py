import tempfile
from pathlib import Path

import pytest
from app.budget import COVERAGE_GATE, CRAWL, Budgets
from app.state import connect


def crawl_shares(budgets: Budgets) -> dict[str, float]:
    return budgets.shares().get(CRAWL, {})


@pytest.fixture
def budgets():
    with tempfile.TemporaryDirectory() as d:
        yield Budgets(connect(str(Path(d) / "b.db")))


def work(budgets, hotkey, assigned, returned, verified):
    budgets.record_coverage(hotkey, assigned, returned)
    if verified:
        budgets.reward(hotkey, "t", verified)


def test_full_coverage_earns_a_share(budgets):
    work(budgets, "a", 100, 100, 100)
    assert crawl_shares(budgets) == {"a": 1.0}


def test_below_the_gate_earns_nothing(budgets):
    work(budgets, "a", 100, 100, 100)
    work(budgets, "omitter", 100, 80, 80)
    shares = crawl_shares(budgets)
    assert "omitter" not in shares
    assert shares == {"a": 1.0}


def test_just_above_the_gate_still_earns(budgets):
    work(budgets, "a", 100, 100, 100)
    work(budgets, "b", 100, 86, 86)
    assert set(crawl_shares(budgets)) == {"a", "b"}


def test_shares_are_proportional_to_verified_work(budgets):
    work(budgets, "big", 1000, 1000, 1000)
    work(budgets, "small", 100, 100, 100)
    shares = crawl_shares(budgets)
    assert shares["big"] == pytest.approx(10 * shares["small"])


def test_a_miner_that_returns_nothing_is_excluded(budgets):
    work(budgets, "a", 100, 100, 100)
    budgets.record_coverage("hoarder", 500, 0)
    assert "hoarder" not in crawl_shares(budgets)
    assert budgets.coverage_report()["hoarder"]["coverage"] == 0.0


def test_gate_is_on_completeness_not_correctness(budgets):
    """Fabricated bodies are the re-fetch's job, not coverage's."""
    work(budgets, "fabricator", 100, 100, 100)
    assert budgets.coverage_report()["fabricator"]["eligible"]
    assert COVERAGE_GATE == 0.85


def test_work_not_yet_decided_does_not_count_against_coverage(budgets):
    budgets.get_or_create("busy")
    assert budgets.coverage_report() == {}

    work(budgets, "busy", 100, 100, 100)
    assert budgets.coverage_report()["busy"]["coverage"] == 1.0
    assert crawl_shares(budgets) == {"busy": 1.0}


def test_a_low_credit_pass_does_not_ramp_the_budget(budgets):
    budgets.reward("a", "t1", 25)
    assert budgets.get_or_create("a").budget == 2
    budgets.reward("a", "t2", 13, ramp=False)
    assert (
        budgets.get_or_create("a").budget,
        budgets.get_or_create("a").verified,
    ) == (2, 38)


def test_peeking_at_an_unknown_miner_stores_nothing(budgets):
    assert budgets.get("stranger").budget == 1
    assert budgets.coverage_report() == {}
    assert "stranger" not in {
        row[0] for row in budgets.db.execute("SELECT hotkey FROM miners")
    }


def test_history_is_newest_first_and_bounded(budgets):
    for n in range(5):
        budgets.reward("a", f"t{n}", 1)
    history = budgets.history("a", limit=3)
    assert [event["task_id"] for event in history] == ["t4", "t3", "t2"]
