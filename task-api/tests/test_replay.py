from __future__ import annotations

import importlib.util
from pathlib import Path

VERIFIER = Path(__file__).resolve().parents[1] / "tools" / "verify_round.py"
spec = importlib.util.spec_from_file_location("verify_round", VERIFIER)
verifier = importlib.util.module_from_spec(spec)
spec.loader.exec_module(verifier)

MANIFEST = [
    {"batch_id": b, "url_count": 10, "urls_hash": "00"} for b in ("a", "b", "c")
]
ORDER = ["a", "b", "c"]


def entry(seq, outcome, task_id=None, hotkey="x"):
    return {
        "seq": seq,
        "outcome": outcome,
        "hotkey": hotkey,
        **({"task_id": task_id} if task_id else {}),
    }


def test_serving_in_order_passes():
    log = [entry(1, "issued", "a"), entry(2, "issued", "b"), entry(3, "issued", "c")]
    ok, why = verifier._replay(MANIFEST, ORDER, log)
    assert ok, why


def test_serving_out_of_order_is_caught():
    ok, why = verifier._replay(MANIFEST, ORDER, [entry(1, "issued", "b")])
    assert not ok
    assert "was next in line" in why


def test_completion_does_not_return_a_batch_to_the_queue():
    log = [entry(1, "issued", "a"), entry(2, "completed", "a"), entry(3, "issued", "b")]
    ok, why = verifier._replay(MANIFEST, ORDER, log)
    assert ok, why


def test_a_reclaimed_batch_returns_to_its_own_position():
    log = [
        entry(1, "issued", "a"),
        entry(2, "issued", "b"),
        entry(3, "reclaimed", "a"),
        entry(4, "issued", "a", hotkey="y"),
        entry(5, "issued", "c"),
    ]
    ok, why = verifier._replay(MANIFEST, ORDER, log)
    assert ok, why


def test_a_reclaimed_batch_may_not_be_jumped():
    log = [
        entry(1, "issued", "a"),
        entry(2, "issued", "b"),
        entry(3, "reclaimed", "a"),
        entry(4, "issued", "c", hotkey="y"),
    ]
    ok, why = verifier._replay(MANIFEST, ORDER, log)
    assert not ok, "a went back to the head and should have been served before c"


def test_a_miner_skips_a_batch_it_held_and_takes_the_next():
    log = [
        entry(1, "issued", "a"),
        entry(2, "reclaimed", "a"),
        entry(3, "issued", "b"),
        entry(4, "issued", "a", hotkey="y"),
    ]
    ok, why = verifier._replay(MANIFEST, ORDER, log)
    assert ok, why


def test_handing_a_miner_a_batch_it_held_before_is_caught():
    log = [entry(1, "issued", "a"), entry(2, "reclaimed", "a"), entry(3, "issued", "a")]
    ok, why = verifier._replay(MANIFEST, ORDER, log)
    assert not ok and "was next in line for x" in why


def test_entries_are_replayed_in_mutation_order_not_write_order():
    out_of_order = [entry(2, "issued", "b"), entry(1, "issued", "a")]
    ok, why = verifier._replay(MANIFEST, ORDER, out_of_order)
    assert ok, why


def test_swapped_urls_are_caught():
    manifest = [
        {
            "batch_id": "a",
            "url_count": 2,
            "urls_hash": verifier.sha256(
                verifier.canonical_json(["https://x.example/1", "https://x.example/2"])
            ),
        }
    ]
    ok, _ = verifier._batches(
        manifest, {"a": ["https://x.example/1", "https://x.example/2"]}
    )
    assert ok
    ok, why = verifier._batches(
        manifest, {"a": ["https://x.example/1", "https://evil.example/2"]}
    )
    assert not ok
    assert "not the ones committed" in why
