"""The log replay a miner uses to check that no batch was passed over without cause."""

from __future__ import annotations

import importlib.util
from pathlib import Path

VERIFIER = Path(__file__).resolve().parents[1] / "tools" / "verify_round.py"
spec = importlib.util.spec_from_file_location("verify_round", VERIFIER)
verifier = importlib.util.module_from_spec(spec)
spec.loader.exec_module(verifier)

MANIFEST = [
    {"batch_id": "a", "hosts": ["h1"], "url_count": 1, "crawl_delay": {"h1": 1.0}},
    {"batch_id": "b", "hosts": ["h1"], "url_count": 1, "crawl_delay": {"h1": 1.0}},
    {"batch_id": "c", "hosts": ["h2"], "url_count": 1, "crawl_delay": {"h2": 1.0}},
]
ORDER = ["a", "b", "c"]


def entry(seq, outcome, task_id=None):
    return {"seq": seq, "outcome": outcome, "hotkey": "x", **({"task_id": task_id} if task_id else {})}


def test_serving_in_order_passes():
    ok, why = verifier._replay(MANIFEST, ORDER, [entry(1, "issued", "a"), entry(2, "issued", "c")])
    assert ok, why


def test_skipping_a_batch_whose_host_is_held_is_allowed():
    """b shares h1 with a, so while a is in flight b must be passed over."""
    log = [entry(1, "issued", "a"), entry(2, "issued", "c")]
    ok, _ = verifier._replay(MANIFEST, ORDER, log)
    assert ok


def test_skipping_a_batch_with_free_hosts_is_caught():
    """c served first while a sits earlier with h1 free -- what favouritism would look like."""
    ok, why = verifier._replay(MANIFEST, ORDER, [entry(1, "issued", "c")])
    assert not ok
    assert "waited with every host free" in why


def test_completed_batches_do_not_return_but_reclaimed_ones_do():
    completed = [entry(1, "issued", "a"), entry(2, "completed", "a"), entry(3, "issued", "b")]
    ok, why = verifier._replay(MANIFEST, ORDER, completed)
    assert ok, why

    reclaimed = [entry(1, "issued", "a"), entry(2, "reclaimed", "a"), entry(3, "issued", "b")]
    ok, why = verifier._replay(MANIFEST, ORDER, reclaimed)
    assert not ok, "a returned to the queue and should have been served before b"


def test_entries_are_replayed_in_mutation_order_not_write_order():
    """Handlers write the log after leaving the atomic section, so arrival order can invert.

    Without the sequence number taken inside the Lua script this replay reports a false violation.
    """
    out_of_order = [
        entry(3, "issued", "b"),
        entry(1, "issued", "a"),
        entry(2, "completed", "a"),
    ]
    ok, why = verifier._replay(MANIFEST, ORDER, out_of_order)
    assert ok, why
