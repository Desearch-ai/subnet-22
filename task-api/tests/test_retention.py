from __future__ import annotations

import sqlite3
import time

from app import rounds
from app.roundstore import RoundStore
from app.validations import Validations


def closed_round(store: RoundStore, urls: int, closed_at: float) -> rounds.Round:
    round_ = rounds.open_round(
        [rounds.Url("example.com", f"https://example.com/{i}") for i in range(urls)],
        100,
    )
    rounds.reveal(round_, "00" * 32)
    store.save(round_)
    store.close(round_.round_id, closed_at)
    return store.get(round_.round_id)


def stored_urls(db: sqlite3.Connection, round_id: str) -> int:
    (batches,) = db.execute(
        "SELECT batches FROM rounds WHERE round_id = ?", (round_id,)
    ).fetchone()
    return batches.count("https://example.com/")


def test_a_round_closed_a_day_ago_keeps_its_manifest_without_its_urls():
    db = sqlite3.connect(":memory:")
    store = RoundStore(db)
    now = time.time()
    old = closed_round(store, 2500, now - 2 * 86_400)
    recent = closed_round(store, 1200, now - 60)

    sealed = store.seal_closed(now - 86_400, 10)

    assert sealed == 1
    assert stored_urls(db, old.round_id) == 0
    assert stored_urls(db, recent.round_id) == 1200
    assert store.get(old.round_id).public_view() == old.public_view()
    assert store.seal_closed(now - 86_400, 10) == 1, (
        "the last sealed round is looked at again, harmlessly"
    )
    assert store.get(old.round_id).public_view() == old.public_view()


def test_a_verdict_drops_its_publish_job_a_day_after_it_was_finalized():
    db = sqlite3.connect(":memory:")
    validations = Validations(db)
    job = {"task_id": "t", "urls": ["https://example.com/a"] * 100}
    for task_id in ("old1", "old2", "new1"):
        validations.finalize(
            task_id, f"submitted/{task_id}", "pass", 100, {**job, "task_id": task_id}
        )
    db.execute(
        "UPDATE final_verdicts SET finalized_at = ? WHERE task_id LIKE 'old%'",
        (time.time() - 2 * 86_400,),
    )
    db.commit()

    validations.drop_publish_copies(time.time() - 86_400, 100)

    kept = {
        task_id: validations.final_verdict(task_id, f"submitted/{task_id}")["publish"]
        for task_id in ("old1", "old2", "new1")
    }
    assert kept["old1"] is None and kept["old2"] is None
    assert kept["new1"]["task_id"] == "new1"
    assert validations.final_verdict("old1", "submitted/old1")["verdict"] == "pass"
