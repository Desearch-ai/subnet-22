from __future__ import annotations

import importlib.util
import inspect
import random
from pathlib import Path

import pytest
from app import proofs

VERIFIER = Path(__file__).resolve().parents[1] / "tools" / "verify_round.py"


def _load_verifier():
    spec = importlib.util.spec_from_file_location("verify_round", VERIFIER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


verifier = _load_verifier()


def _manifest(n: int) -> list[dict]:
    return [
        {"batch_id": f"b{i:04d}", "url_count": 10 + i, "urls_hash": f"{i:064x}"}
        for i in range(n)
    ]


def test_manifest_hash_ignores_our_packing_order():
    entries = _manifest(50)
    shuffled = entries[:]
    random.shuffle(shuffled)
    assert proofs.manifest_hash(entries, 100) == proofs.manifest_hash(shuffled, 100)


def test_manifest_hash_notices_any_content_change():
    entries = _manifest(50)
    tampered = [dict(entries[0], url_count=999)] + entries[1:]
    assert proofs.manifest_hash(entries, 100) != proofs.manifest_hash(tampered, 100)


def test_serve_order_is_a_permutation_and_not_the_input_order():
    ids = [f"b{i:04d}" for i in range(200)]
    order = proofs.serve_order("seed", ids)
    assert sorted(order) == sorted(ids)
    assert order != ids


def test_serve_order_does_not_depend_on_input_order():
    ids = [f"b{i:04d}" for i in range(200)]
    assert proofs.serve_order("s", ids) == proofs.serve_order("s", list(reversed(ids)))


def test_a_different_seed_gives_a_different_order():
    ids = [f"b{i:04d}" for i in range(200)]
    assert proofs.serve_order("seed-a", ids) != proofs.serve_order("seed-b", ids)


def test_the_rule_takes_no_miner_identity():
    for name in ("manifest_hash", "position_key", "serve_order", "merkle_root"):
        params = set(inspect.signature(getattr(proofs, name)).parameters)
        assert not params & {"hotkey", "miner", "uid", "caller"}, name


def test_merkle_root_is_sensitive_to_every_leaf():
    leaves = [f"line-{i}".encode() for i in range(9)]
    root = proofs.merkle_root(leaves)
    for i in range(len(leaves)):
        altered = leaves[:]
        altered[i] = b"tampered"
        assert proofs.merkle_root(altered) != root, f"leaf {i} did not affect the root"


def test_merkle_root_of_nothing_is_defined():
    assert proofs.merkle_root([]) == proofs.sha256(b"")


@pytest.mark.parametrize("size", [0, 1, 2, 3, 8, 9, 64, 65])
def test_verifier_merkle_matches_server(size):
    leaves = [f"entry-{i}".encode() for i in range(size)]
    assert verifier.merkle_root(leaves) == proofs.merkle_root(leaves)


def test_verifier_manifest_hash_matches_server():
    entries = _manifest(37)
    assert verifier.manifest_hash(entries, 100) == proofs.manifest_hash(entries, 100)


def test_verifier_serve_order_matches_server():
    ids = [f"b{i:04d}" for i in range(300)]
    for seed in ("0xdead", "0xbeef", "abc"):
        assert verifier.serve_order(seed, ids) == proofs.serve_order(seed, ids)


def test_verifier_algorithm_label_matches_server():
    assert verifier.ALGORITHM == proofs.ALGORITHM


def test_changing_the_seed_block_changes_the_commitment():
    """Else we could shop for a block with a favourable hash."""
    entries = _manifest(20)
    assert proofs.manifest_hash(entries, 100) != proofs.manifest_hash(entries, 101)


def test_a_batch_mixes_hosts_so_one_blocked_site_costs_one_row():
    from app.rounds import Url, pack

    urls = [
        Url(host=f"site{h}.example", url=f"https://site{h}.example/{n}")
        for h in range(5)
        for n in range(10)
    ]
    batches = pack(urls, batch_target=10)

    assert len(batches) == 5
    assert all(len({u.host for u in batch.urls}) == 5 for batch in batches)
    assert sorted(u.url for batch in batches for u in batch.urls) == sorted(
        u.url for u in urls
    )


def test_packing_keeps_every_url_when_hosts_are_lopsided():
    from app.rounds import Url, pack

    urls = [Url(host="big.example", url=f"https://big.example/{n}") for n in range(25)]
    urls += [Url(host="small.example", url="https://small.example/1")]
    batches = pack(urls, batch_target=10)

    assert sum(len(b.urls) for b in batches) == 26
    assert batches[0].urls[1].host == "small.example"
