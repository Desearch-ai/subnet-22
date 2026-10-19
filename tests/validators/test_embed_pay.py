import sqlite3

import pytest

from desearch.kinds import EMBED
from neurons.validators.ledger import Ledger
from tests.validators.test_crawl_pay import NOW, logged
from tests.validators.test_crawl_pay import checked as crawl_checked
from tests.validators.test_crawl_pay import ledger_with as crawl_ledger_with


def embedded(key: str, miner: str, at: float, chars: int = 1000) -> dict:
    return {
        "kind": "embed",
        "key": key,
        "task_id": key,
        "miner": miner,
        "completed_at": at,
        "texts": 10,
        "chars": chars,
    }


def ledger_with(uploads: int, chars=lambda n: 1000, ledger=None) -> Ledger:
    """A miner's embed uploads, a minute apart."""
    ledger = ledger or Ledger(":memory:")
    ledger.add_uploads(
        [embedded(f"e{n}", "m", NOW - 3600 + 60 * n, chars(n)) for n in range(uploads)]
    )
    return ledger


def checked(ledger: Ledger, n: int, verdict: str, chars: int = 1000):
    ledger.record_check(
        f"e{n}",
        "m",
        NOW - 3600 + 60 * n,
        verdict,
        chars if verdict == "pass" else 0,
        chars,
        chars,
        chars,
        at=NOW - 10 + n / 100,
        kind=EMBED,
    )


def test_unchecked_embed_uploads_are_paid_their_characters_at_the_rate_the_checks_paid():
    ledger = ledger_with(10, chars=lambda n: 1000 + 10 * n)
    checked(ledger, 0, "pass", 1000)
    checked(ledger, 5, "pass", 1050)

    found = ledger.embed_paid(NOW)["m"]

    assert found.rate == 1.0
    assert found.amount == pytest.approx(sum(1000 + 10 * n for n in range(10)))
    assert (found.uploads, found.checked, found.failed) == (10, 2, 0)


def test_a_failed_embed_check_takes_back_what_came_since_the_last_pass_and_costs_its_characters():
    ledger = ledger_with(10)
    checked(ledger, 2, "pass")
    checked(ledger, 6, "fail")

    found = ledger.embed_paid(NOW)["m"]

    kept = [0, 1, 7, 8, 9]
    assert found.rate == 0.5
    assert found.amount == pytest.approx(1000 + 0.5 * 1000 * len(kept) - 1000)


def test_two_failed_embed_checks_among_the_last_ten_take_back_the_day_before():
    ledger = ledger_with(10)
    checked(ledger, 1, "fail")
    checked(ledger, 4, "pass")
    checked(ledger, 8, "fail")

    found = ledger.embed_paid(NOW)["m"]

    assert found.failed == 2
    assert found.amount == pytest.approx(1000 / 3000 * 1000 - 2000)


def test_embed_work_leaves_crawl_pay_as_it_was():
    ledger = crawl_ledger_with(10)
    crawl_checked(ledger, 2, "pass", credited=100)
    crawl_checked(ledger, 6, "fail", credited=0)
    before = ledger.crawl_paid(NOW)

    ledger_with(10, ledger=ledger)
    checked(ledger, 1, "fail")
    checked(ledger, 3, "fail")

    assert ledger.crawl_paid(NOW) == before
    assert ledger.embed_paid(NOW)["m"].amount == pytest.approx(-2000)


def test_an_upload_log_entry_without_a_kind_is_a_crawl_upload():
    ledger = Ledger(":memory:")
    old = logged("c1", "m", NOW)
    ledger.add_uploads([old, embedded("e1", "m", NOW, chars=777)])

    assert ledger.db.execute(
        "SELECT key, kind, chars, assigned FROM uploads ORDER BY key"
    ).fetchall() == [("c1", "crawl", 0, 100), ("e1", "embed", 777, 10)]


OLD_LEDGER = """
CREATE TABLE uploads (
    key TEXT PRIMARY KEY, task_id TEXT NOT NULL, miner TEXT NOT NULL,
    completed_at REAL NOT NULL, assigned INTEGER NOT NULL, ok INTEGER NOT NULL,
    errors INTEGER NOT NULL
);
CREATE TABLE checks (
    key TEXT PRIMARY KEY, miner TEXT NOT NULL, completed_at REAL NOT NULL,
    scored_at REAL NOT NULL, verdict TEXT NOT NULL, credited INTEGER NOT NULL,
    returned INTEGER NOT NULL, content INTEGER NOT NULL, assigned INTEGER NOT NULL
);
INSERT INTO uploads VALUES ('u0', 'u0', 'm', 996400.0, 100, 100, 0);
INSERT INTO uploads VALUES ('u1', 'u1', 'm', 996460.0, 100, 100, 0);
INSERT INTO checks VALUES ('u0', 'm', 996400.0, 999990.0, 'pass', 90, 100, 100, 100);
"""


def test_a_ledger_from_before_embed_uploads_were_logged_keeps_its_rows_as_crawl(
    tmp_path,
):
    path = tmp_path / "verdicts.db"
    old = sqlite3.connect(path)
    old.executescript(OLD_LEDGER)
    old.commit()
    old.close()

    ledger = Ledger(path)
    found = ledger.crawl_paid(NOW)["m"]
    ledger.add_uploads([embedded("e1", "m", NOW - 60)])
    ledger.close()

    assert (found.amount, found.uploads, found.checked) == (pytest.approx(180), 2, 1)
    reopened = Ledger(path)
    assert reopened.db.execute(
        "SELECT key, kind, chars FROM uploads ORDER BY key"
    ).fetchall() == [("e1", "embed", 1000), ("u0", "crawl", 0), ("u1", "crawl", 0)]
    assert reopened.db.execute("SELECT kind FROM checks").fetchall() == [("crawl",)]
    assert set(reopened.embed_paid(NOW)) == {"m"}
