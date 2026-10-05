import sqlite3

from app.rounds import Batch, Round, Url
from app.roundstore import RoundStore


def test_open_rounds_are_found_through_their_own_indexes():
    store = RoundStore(sqlite3.connect(":memory:"))
    batch = Batch("b", [Url("site.example", "https://site.example/")], {})
    for n, (seed, filled, closed) in enumerate(
        [(None, None, None), ("s", None, None), ("s", 1.0, None), ("s", 1.0, 2.0)]
    ):
        store.save(Round(f"r{n}", {"b": batch}, "h", 1, float(n), seed=seed))
        if filled:
            store.mark_filled(f"r{n}", filled)
        if closed:
            store.close(f"r{n}", closed)

    assert [r.round_id for r in store.unfilled()] == ["r1"]
    assert store.open_revealed() == ["r2"]
