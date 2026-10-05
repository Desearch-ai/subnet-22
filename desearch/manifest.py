"""The signed notes the task API leaves in the uploads bucket: one beside every frozen upload, and the log of every upload."""

from __future__ import annotations

import hashlib
import json

from bittensor.wallets import Keypair

OPEN_LIST_KEY = "validation/open.json"
UPLOAD_LOG_LATEST = "log/uploads/latest.json"
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


def upload_log_key(seq: int) -> str:
    return f"log/uploads/seq/{seq:012d}.json"


def log_payload(body: dict) -> bytes:
    unsigned = {name: value for name, value in body.items() if name != "signature"}
    return json.dumps(
        unsigned, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode()


def verify_log(signed: dict, signer: str) -> bool:
    try:
        return signed["signer"] == signer and Keypair(ss58_address=signer).verify(
            log_payload(signed), bytes.fromhex(signed["signature"])
        )
    except Exception:
        return False


def seed_from_hash(block_hash: str) -> str:
    return hashlib.sha256(str(block_hash).encode()).hexdigest()


def local_block_hash(block: int) -> str:
    return f"local:{block}"
