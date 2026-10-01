import tempfile
from pathlib import Path

import pytest
from app.budget import CEILING, CRAWL, Budgets
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


def test_short_tasks_are_paid_for_what_came_back(budgets):
    work(budgets, "a", 100, 100, 100)
    work(budgets, "omitter", 100, 80, 80)
    assert crawl_shares(budgets) == pytest.approx({"a": 100 / 180, "omitter": 80 / 180})


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


def test_a_bad_task_takes_its_urls_back_from_the_day(budgets):
    work(budgets, "a", 3000, 3000, 3000)
    work(budgets, "b", 1000, 1000, 1000)
    budgets.credit("a", -1000)
    assert crawl_shares(budgets) == {"a": 2 / 3, "b": 1 / 3}


def test_the_budget_grows_by_half_with_each_pass_up_to_the_ceiling(budgets):
    grown = []
    for n in range(14):
        grown.append(budgets.reward("a", f"t{n}", 1).budget)
    assert grown == [2, 3, 4, 6, 9, 13, 19, 28, 42, 63, 94, CEILING, CEILING, CEILING]
    assert budgets.penalise("a", "t99", "verification_failed").budget == CEILING // 2


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
