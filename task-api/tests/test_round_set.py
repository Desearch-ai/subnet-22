"""tools/verify_round.py on rounds that commit to their URL set and are cut by their seed (desearch-round-3)."""

from __future__ import annotations

import base64
import importlib.util
import json
import zlib
from pathlib import Path

VERIFIER = Path(__file__).resolve().parents[1] / "tools" / "verify_round.py"
spec = importlib.util.spec_from_file_location("verify_round", VERIFIER)
verifier = importlib.util.module_from_spec(spec)
spec.loader.exec_module(verifier)

PAIRS = (
    [["a.example", f"https://a.example/{n}"] for n in range(5)]
    + [["b.example", f"https://b.example/{n}"] for n in range(3)]
    + [["c.example", "https://c.example/0"]]
)


def test_the_cut_matches_the_task_api():
    # The same vectors are asserted in task-api/src/rounds.rs.
    assert (
        verifier.set_hash(PAIRS)
        == "a2315215c5ad3ef308edb903a7a7bbee6b987b971d557b36bcb21e764b064332"
    )
    assert (
        verifier.set_manifest_hash("r1", 77, 4, 9, verifier.set_hash(PAIRS))
        == "3ebc415543cf7d37d74b7a1043ebd7db706b685a2717047212aa974cd2eca0a6"
    )
    cut = [
        (batch, [url for _, url in pairs])
        for batch, pairs in verifier.cut(PAIRS, "seed", "r1", 4)
    ]
    assert cut == [
        (
            "34af69571cea3b51",
            ["https://c.example/0", "https://a.example/0", "https://b.example/1"],
        ),
        (
            "5388cbf670e889e5",
            ["https://a.example/3", "https://b.example/0", "https://a.example/1"],
        ),
        (
            "7006229857aa481c",
            ["https://a.example/2", "https://b.example/2", "https://a.example/4"],
        ),
    ]


def test_the_order_the_bot_sent_urls_in_plays_no_part():
    assert verifier.cut(list(reversed(PAIRS)), "seed", "r1", 4) == verifier.cut(
        PAIRS, "seed", "r1", 4
    )
    assert verifier.set_hash(list(reversed(PAIRS))) == verifier.set_hash(PAIRS)
    assert verifier.cut(PAIRS, "other", "r1", 4) != verifier.cut(PAIRS, "seed", "r1", 4)


def published(pairs=PAIRS, seed="seed", round_id="r1", task_urls=4, seed_block=77):
    """A closed round as GET /v1/rounds/<id> shows it."""
    urls_hash = verifier.set_hash(pairs)
    manifest = [
        {
            "batch_id": batch,
            "url_count": len(batch_pairs),
            "urls_hash": verifier.sha256(
                verifier.canonical_json([url for _, url in batch_pairs])
            ),
        }
        for batch, batch_pairs in verifier.cut(pairs, seed, round_id, task_urls)
    ]
    view = {
        "round_id": round_id,
        "algorithm": verifier.SET_ALGORITHM,
        "seed_block": seed_block,
        "task_urls": task_urls,
        "url_count": len(pairs),
        "urls_hash": urls_hash,
        "manifest_hash": verifier.set_manifest_hash(
            round_id, seed_block, task_urls, len(pairs), urls_hash
        ),
    }
    return view, manifest


def test_a_round_cut_from_its_committed_set_verifies():
    view, manifest = published()
    assert verifier._set_commitment(view, manifest, "seed", PAIRS)
    assert verifier._set_commitment(
        view, manifest, "seed", None
    ), "without the set, the commitment and the batch names still check"


def test_a_url_moved_to_another_batch_is_caught():
    view, manifest = published()
    moved = [dict(entry) for entry in manifest]
    moved[0]["urls_hash"], moved[1]["urls_hash"] = (
        moved[1]["urls_hash"],
        moved[0]["urls_hash"],
    )
    assert not verifier._set_commitment(view, moved, "seed", PAIRS)


def test_a_set_changed_after_the_commitment_is_caught():
    view, manifest = published()
    swapped = PAIRS[:-1] + [["d.example", "https://d.example/0"]]
    assert not verifier._set_commitment(view, manifest, "seed", swapped)
    tampered = dict(view, urls_hash=verifier.set_hash(swapped))
    assert not verifier._set_commitment(tampered, manifest, "seed", None)


def test_batches_named_by_another_seed_are_caught():
    view, manifest = published(seed="chosen")
    assert not verifier._set_commitment(view, manifest, "seed", None)


def test_the_round_urls_are_read_as_the_pages_bucket_keeps_them(tmp_path):
    lists = {batch: pairs for batch, pairs in verifier.cut(PAIRS, "seed", "r1", 4)}
    packed = "z:" + base64.b64encode(zlib.compress(json.dumps(lists).encode())).decode()
    (tmp_path / "packed").write_text(packed)
    (tmp_path / "plain.json").write_text(json.dumps(PAIRS))
    assert sorted(verifier.read_urls(tmp_path / "packed")) == sorted(PAIRS)
    assert verifier.read_urls(tmp_path / "plain.json") == PAIRS


def test_the_signer_and_the_anchor_default_to_the_pinned_ones(monkeypatch):
    import sys

    from desearch.trust import TASK_API_ANCHOR_HOTKEY, TASK_API_SIGNER

    given = []
    monkeypatch.setattr(verifier, "verify", lambda *args: given.append(args) or True)
    monkeypatch.setattr(sys, "argv", ["verify_round.py", "--round", "r1"])
    assert verifier.main() == 0
    assert (given[0][4], given[0][7]) == (TASK_API_SIGNER, TASK_API_ANCHOR_HOTKEY)
    monkeypatch.setattr(
        sys, "argv", ["verify_round.py", "--round", "r1", "--signer", "5Other"]
    )
    verifier.main()
    assert given[1][4] == "5Other"
