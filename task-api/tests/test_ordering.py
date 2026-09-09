"""The distribution rule, and the guarantee that the public verifier agrees with it."""

from __future__ import annotations

import importlib.util
import inspect
import random
from pathlib import Path

import pytest

from app import ordering

VERIFIER = Path(__file__).resolve().parents[1] / "tools" / "verify_round.py"


def _load_verifier():
    """Load the miner-facing script, which reimplements the rule independently."""
    spec = importlib.util.spec_from_file_location("verify_round", VERIFIER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


verifier = _load_verifier()


def _manifest(n: int) -> list[dict]:
    return [
        {
            "batch_id": f"b{i:04d}",
            "hosts": [f"h{i % 7}.example"],
            "url_count": 10 + i,
            "crawl_delay": {f"h{i % 7}.example": 1.0},
        }
        for i in range(n)
    ]


# --- the rule itself -------------------------------------------------------


def test_manifest_hash_ignores_our_packing_order():
    entries = _manifest(50)
    shuffled = entries[:]
    random.shuffle(shuffled)
    assert ordering.manifest_hash(entries) == ordering.manifest_hash(shuffled)


def test_manifest_hash_notices_any_content_change():
    entries = _manifest(50)
    tampered = [dict(entries[0], url_count=999)] + entries[1:]
    assert ordering.manifest_hash(entries) != ordering.manifest_hash(tampered)


def test_serve_order_is_a_permutation_and_not_the_input_order():
    ids = [f"b{i:04d}" for i in range(200)]
    order = ordering.serve_order("seed", ids)
    assert sorted(order) == sorted(ids)
    assert order != ids


def test_serve_order_does_not_depend_on_input_order():
    ids = [f"b{i:04d}" for i in range(200)]
    assert ordering.serve_order("s", ids) == ordering.serve_order(
        "s", list(reversed(ids))
    )


def test_a_different_seed_gives_a_different_order():
    ids = [f"b{i:04d}" for i in range(200)]
    assert ordering.serve_order("seed-a", ids) != ordering.serve_order("seed-b", ids)


def test_the_rule_takes_no_miner_identity():
    """If the ordering functions cannot see a hotkey, they cannot favour one."""
    for name in ("manifest_hash", "position_key", "serve_order", "merkle_root"):
        params = set(inspect.signature(getattr(ordering, name)).parameters)
        assert not params & {"hotkey", "miner", "uid", "caller"}, name


def test_merkle_root_is_sensitive_to_every_leaf():
    leaves = [f"line-{i}".encode() for i in range(9)]
    root = ordering.merkle_root(leaves)
    for i in range(len(leaves)):
        altered = leaves[:]
        altered[i] = b"tampered"
        assert ordering.merkle_root(altered) != root, (
            f"leaf {i} did not affect the root"
        )


def test_merkle_root_of_nothing_is_defined():
    assert ordering.merkle_root([]) == ordering.sha256(b"")


# --- the verifier must not drift from the server ---------------------------


@pytest.mark.parametrize("size", [0, 1, 2, 3, 8, 9, 64, 65])
def test_verifier_merkle_matches_server(size):
    leaves = [f"entry-{i}".encode() for i in range(size)]
    assert verifier.merkle_root(leaves) == ordering.merkle_root(leaves)


def test_verifier_manifest_hash_matches_server():
    entries = _manifest(37)
    assert verifier.manifest_hash(entries) == ordering.manifest_hash(entries)


def test_verifier_serve_order_matches_server():
    ids = [f"b{i:04d}" for i in range(300)]
    for seed in ("0xdead", "0xbeef", "abc"):
        assert verifier.serve_order(seed, ids) == ordering.serve_order(seed, ids)


def test_verifier_algorithm_label_matches_server():
    assert verifier.ALGORITHM == ordering.ALGORITHM
