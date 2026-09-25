import time

import pytest
from app.budget import CRAWL, HOUR, SHARE_WINDOW_H, Budgets
from app.state import connect


def crawl_shares(budgets: Budgets) -> dict[str, float]:
    return budgets.shares().get(CRAWL, {})


def earned(budgets: Budgets, hotkey: str, urls: int, hours_ago: int = 0) -> None:
    budgets.record_coverage(hotkey, urls, urls)
    budgets.reward(hotkey, f"task-{hotkey}-{hours_ago}", urls, ramp=False)
    if hours_ago:
        budgets.db.execute(
            "UPDATE credits SET hour = hour - ? WHERE hotkey = ? AND hour = ?",
            (hours_ago, hotkey, int(time.time() // HOUR)),
        )
        budgets.db.commit()


def test_shares_split_the_windows_verified_pages(tmp_path):
    budgets = Budgets(connect(str(tmp_path / "b.db")))
    earned(budgets, "a", 300)
    earned(budgets, "b", 100)

    assert crawl_shares(budgets) == {"a": 0.75, "b": 0.25}


def test_work_older_than_the_window_no_longer_pays(tmp_path):
    budgets = Budgets(connect(str(tmp_path / "b.db")))
    earned(budgets, "old", 1000, hours_ago=SHARE_WINDOW_H + 1)
    earned(budgets, "new", 10)

    assert crawl_shares(budgets) == {"new": 1.0}
    assert budgets.get_or_create("old").verified == 1000


def test_a_miner_under_the_coverage_gate_earns_no_share(tmp_path):
    budgets = Budgets(connect(str(tmp_path / "b.db")))
    earned(budgets, "full", 100)
    budgets.record_coverage("thin", 100, 10)
    budgets.reward("thin", "t", 10, ramp=False)

    assert crawl_shares(budgets) == {"full": 1.0}


def test_pruning_keeps_a_week_of_credit(tmp_path):
    budgets = Budgets(connect(str(tmp_path / "b.db")))
    earned(budgets, "a", 5, hours_ago=SHARE_WINDOW_H * 7 + 5)
    earned(budgets, "b", 5, hours_ago=2)
    budgets.prune()

    assert [
        r[0] for r in budgets.db.execute("SELECT hotkey FROM credits").fetchall()
    ] == ["b"]


def test_nothing_verified_means_no_shares(tmp_path):
    assert Budgets(connect(str(tmp_path / "b.db"))).shares() == {}


def test_coverage_is_judged_over_the_same_window_as_credit(tmp_path):
    budgets = Budgets(connect(str(tmp_path / "b.db")))
    budgets.record_coverage("recovered", 750, 0)
    budgets.db.execute("UPDATE coverage SET hour = hour - ?", (SHARE_WINDOW_H + 1,))
    budgets.db.commit()
    earned(budgets, "recovered", 100)

    assert crawl_shares(budgets) == {"recovered": 1.0}, (
        "an outage a day ago no longer shuts it out"
    )

    budgets.record_coverage("recovered", 750, 0)
    assert crawl_shares(budgets) == {}, "the same outage inside the window does"


def test_each_pool_is_shared_out_on_its_own(tmp_path):
    budgets = Budgets(connect(str(tmp_path / "b.db")))
    earned(budgets, "crawler", 300)
    budgets.record_coverage("hoarder", 1000, 0)
    for hotkey, amount in (("embedder", 40), ("second", 10), ("hoarder", 10)):
        budgets.reward(hotkey, f"t-{hotkey}", amount, ramp=False, pool="embed")

    shares = budgets.shares()

    assert shares[CRAWL] == {"crawler": 1.0}
    assert shares["embed"] == pytest.approx(
        {"embedder": 4 / 6, "second": 1 / 6, "hoarder": 1 / 6}
    ), "the crawl coverage gate does not apply to other pools"
