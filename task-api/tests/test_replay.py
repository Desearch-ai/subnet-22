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


def test_a_batch_dropped_as_it_was_drawn_leaves_the_queue():
    log = [entry(1, "dropped", "a", hotkey=""), entry(2, "issued", "b"), entry(3, "issued", "c")]
    ok, why = verifier._replay(MANIFEST, ORDER, log)
    assert ok, why


def test_a_batch_dropped_after_its_last_attempt_changes_nothing():
    log = [entry(1, "issued", "a"), entry(2, "dropped", "a"), entry(3, "issued", "b")]
    ok, why = verifier._replay(MANIFEST, ORDER, log)
    assert ok, why


def test_only_the_round_s_own_receipts_are_checked():
    log = [{"seq": 1, "outcome": "issued", "hotkey": "x", "task_id": "a", "requested_at": 1.0}]
    other = {"body": {"round_id": "other", "seq": 9, "outcome": "issued"}, "signature": "00"}
    ok, why = verifier._receipts("r1", [other], log, "signer")
    assert ok, why


def test_a_batch_taken_for_a_miner_never_told_of_it_holds_up_no_other():
    never = {**entry(2, "reclaimed", "a"), "cause": "never_issued"}
    log = [entry(1, "issued", "b"), never, entry(3, "issued", "a", hotkey="y"), entry(4, "issued", "c")]
    ok, why = verifier._replay(MANIFEST, ORDER, log)
    assert ok, why


def test_a_batch_never_told_of_counts_as_held_by_its_miner_once_it_is_back():
    never = {**entry(1, "reclaimed", "a"), "cause": "never_issued"}
    log = [never, entry(2, "issued", "b"), entry(3, "issued", "a", hotkey="y")]
    ok, why = verifier._replay(MANIFEST, ORDER, log)
    assert ok, why


def test_a_batch_issued_then_found_never_told_of_goes_back_to_its_place():
    never = {**entry(2, "reclaimed", "a"), "cause": "never_issued"}
    log = [entry(1, "issued", "a"), never, entry(3, "issued", "a", hotkey="y"), entry(4, "issued", "b")]
    ok, why = verifier._replay(MANIFEST, ORDER, log)
    assert ok, why


def test_one_signed_line_excuses_a_batch_only_until_it_is_back():
    never = {**entry(1, "reclaimed", "a", hotkey="z"), "cause": "never_issued"}
    ok, why = verifier._replay(MANIFEST, ORDER, [never, entry(2, "issued", "b")])
    assert not ok
    assert "was next in line" in why


def test_a_batch_said_to_be_never_told_of_was_still_issued_in_its_turn():
    never = {**entry(2, "reclaimed", "c"), "cause": "never_issued"}
    ok, why = verifier._replay(MANIFEST, ORDER, [entry(1, "issued", "c"), never])
    assert not ok
    assert "was next in line" in why


def test_only_a_batch_never_told_of_counts_as_held_with_no_line_to_show_it():
    expired = {**entry(1, "reclaimed", "a", hotkey="y"), "cause": "expired"}
    log = [expired, entry(2, "issued", "b", hotkey="y"), entry(3, "issued", "a", hotkey="z")]
    ok, why = verifier._replay(MANIFEST, ORDER, log)
    assert not ok
    assert "was next in line" in why


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


def issued(seq, waited_from, served_at, hotkey="x"):
    return {
        "seq": seq,
        "outcome": "issued",
        "hotkey": hotkey,
        "waited_from": waited_from,
        "served_at": served_at,
    }


def test_tasks_given_in_the_order_places_were_taken_pass():
    ok, why = verifier._line_order([issued(1, 10.0, 100.0), issued(2, 20.0, 101.0)])
    assert ok, why


def test_a_place_passed_while_its_miner_had_not_asked_yet_passes():
    ok, why = verifier._line_order([issued(1, 20.0, 100.0), issued(2, 10.0, 109.0)])
    assert ok, why


def test_a_place_passed_once_is_served_next_however_long_the_next_task_takes():
    ok, why = verifier._line_order([issued(1, 20.0, 100.0), issued(2, 10.0, 150.0)])
    assert ok, why


def test_a_place_passed_for_longer_than_it_lasts_is_caught():
    log = [issued(1, 20.0, 100.0), issued(2, 30.0, 150.0), issued(3, 10.0, 170.0)]
    ok, why = verifier._line_order(log)
    assert not ok
    assert "passed for 50.0s" in why


def test_a_later_place_served_with_the_same_release_is_no_second_pass():
    log = [issued(1, 90.0, 100.0), issued(2, 110.0, 150.0), issued(3, 80.0, 150.01)]
    ok, why = verifier._line_order(log)
    assert ok, why


def test_tasks_given_outside_the_line_are_counted_in_the_open():
    ok, why = verifier._line_order([issued(1, 10.0, 100.0), entry(2, "issued", "a")])
    assert ok
    assert "1 outside it" in why


def test_a_place_passed_and_gone_counts_only_while_it_waited():
    log = [issued(1, 20.0, 100.0), issued(2, 10.0, 101.0), issued(3, 30.0, 150.0)]
    ok, why = verifier._line_order(log)
    assert ok, why


def test_a_log_from_before_the_line_has_nothing_to_check():
    ok, why = verifier._line_order([entry(1, "issued", "a"), entry(2, "issued", "b")])
    assert ok, why


def test_the_line_tool_reads_the_public_events_oldest_first(monkeypatch):
    spec = importlib.util.spec_from_file_location(
        "verify_line", VERIFIER.with_name("verify_line.py")
    )
    tool = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tool)
    pages = {
        "/v1/events?outcome=issued&limit=100": {
            "events": [{"id": 3}, {"id": 2}],
            "next": 2,
        },
        "/v1/events?outcome=issued&limit=100&before=2": {
            "events": [{"id": 1}],
            "next": None,
        },
    }
    monkeypatch.setattr(tool.verify_round, "get", lambda api, path: pages[path])
    assert [event["id"] for event in tool.issues("api", 10)] == [1, 2, 3]
