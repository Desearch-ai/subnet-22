"""The rule that decides serve order. Reimplemented in tools/verify_round.py."""

from __future__ import annotations

import hashlib
import json

ALGORITHM = "desearch-serve-order-1"


def canonical(value) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode()


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def manifest_hash(entries: list[dict]) -> str:
    ordered = sorted(entries, key=lambda entry: entry["batch_id"])
    return sha256(canonical({"algorithm": ALGORITHM, "batches": ordered}))


def position_key(seed: str, batch_id: str) -> str:
    return sha256(f"{seed}:{batch_id}".encode())


def serve_order(seed: str, batch_ids: list[str]) -> list[str]:
    return sorted(
        batch_ids, key=lambda batch_id: (position_key(seed, batch_id), batch_id)
    )


def merkle_root(leaves: list[bytes]) -> str:
    if not leaves:
        return sha256(b"")
    level = [hashlib.sha256(leaf).digest() for leaf in leaves]
    while len(level) > 1:
        if len(level) % 2:
            level.append(level[-1])
        level = [
            hashlib.sha256(level[i] + level[i + 1]).digest() for i in range(0, len(level), 2)
        ]
    return level[0].hex()


def log_leaf(entry: dict) -> bytes:
    return canonical(entry)
