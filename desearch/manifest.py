"""The signed note the task API leaves next to every frozen upload, and the sample seed."""

from __future__ import annotations

import hashlib
import json

try:
    from bittensor_wallet import Keypair
except ImportError:
    from bittensor import Keypair

OPEN_LIST_KEY = "validation/open.json"
# Far enough ahead to anchor the commitment before the seed block.
REVEAL_AFTER_BLOCKS = 10
FIELDS = (
    "task_id",
    "kind",
    "round_id",
    "miner",
    "key",
    "urls",
    "size",
    "etag",
    "frozen_block",
    "seed_block",
    "completed_at",
    "deadline",
    "model",
    "texts",
    "chars",
    "input_key",
    "input_sha256",
)


def payload(manifest: dict) -> bytes:
    body = {name: manifest[name] for name in FIELDS if manifest.get(name) is not None}
    return json.dumps(
        body, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode()


def verify(manifest: dict, signer: str) -> bool:
    try:
        return Keypair(ss58_address=signer).verify(
            payload(manifest), bytes.fromhex(manifest["signature"])
        )
    except Exception:
        return False


def seed_from_hash(block_hash: str) -> str:
    return hashlib.sha256(str(block_hash).encode()).hexdigest()


def local_block_hash(block: int) -> str:
    return f"local:{block}"
