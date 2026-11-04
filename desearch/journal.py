"""The task API's hash-chained log: every entry names the hash of the one before, the files it is published in are signed, and the
head is anchored on chain, so a reader can tell an altered, dropped or forked log from the one the API committed to.

The same checks are in the API's Rust (`task-api/src/journal.rs`); the bytes hashed and signed are Python's own `json.dumps`.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Callable, Iterable, Optional

GENESIS = "0" * 64
LATEST_KEY = "log/journal/latest.json"
# The commitment field that names an anchor, followed by the entry number; the head is a Sha256 field beside it.
ANCHOR_TAG = "sn22j1:"


def segment_key(segment: int) -> str:
    return f"log/journal/seq/{segment:012d}.json"


def blob_key(sha256: str) -> str:
    return f"log/journal/blobs/{sha256}"


def canonical(value) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode()


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def hash_of(entry: dict) -> str:
    return sha256(canonical(entry))


def signing_payload(
    method: str, path: str, body: bytes, timestamp: str, nonce: str
) -> bytes:
    return f"{method}\n{path}\n{sha256(body)}\n{timestamp}\n{nonce}".encode()


def request_hash(
    method: str,
    path: str,
    body: bytes,
    timestamp: str,
    nonce: str,
    hotkey: str,
    signature: str,
) -> str:
    """One signed request's identity, as the API logs it and a relaying validator keeps it."""
    return sha256(
        signing_payload(method, path, body, timestamp, nonce)
        + f"\n{hotkey}\n{signature.lower()}".encode()
    )


class Broken(Exception):
    """The log read does not hold together; the message says where."""


def verify_chain(last: int, prev: str, entries: Iterable[dict]) -> tuple[int, str]:
    """Checks that `entries` follow on from entry `last` with hash `prev`; returns the new last number and head."""
    for entry in entries:
        seq = entry.get("seq")
        if seq != last + 1:
            raise Broken(f"entry {seq} where {last + 1} was due: entries are missing")
        if entry.get("prev") != prev:
            raise Broken(
                f"entry {seq} names {entry.get('prev')} as the one before it, which hashes to {prev}"
            )
        last, prev = seq, hash_of(entry)
    return last, prev


def signed_payload(body: dict) -> bytes:
    return canonical(
        {name: value for name, value in body.items() if name != "signature"}
    )


def signed_by(body: dict, signer: str) -> bool:
    try:
        from bittensor.wallets import Keypair

        return body.get("signer") == signer and Keypair(ss58_address=signer).verify(
            signed_payload(body), bytes.fromhex(body["signature"])
        )
    except Exception:
        return False


def verify_segment(segment: dict, signer: str, last: int, prev: str) -> tuple[int, str]:
    """One published file: signed by `signer`, chained from `(last, prev)`, its range and head those of its entries."""
    number = segment.get("segment")
    if not signed_by(segment, signer):
        raise Broken(f"file {number} is not signed by {signer}")
    if segment.get("prev") != prev:
        raise Broken(f"file {number} follows {segment.get('prev')}, not {prev}")
    end, head = verify_chain(last, prev, segment.get("entries") or [])
    if segment.get("head") != head:
        raise Broken(
            f"file {number} names head {segment.get('head')}, its entries hash to {head}"
        )
    if segment.get("first") != last + 1 or segment.get("last") != end:
        raise Broken(
            f"file {number} names entries {segment.get('first')}..{segment.get('last')}, holds {last + 1}..{end}"
        )
    return end, head


def blobs_named(entries: Iterable[dict]) -> set[str]:
    """The blobs entries name, which a reader fetches beside the file."""
    found: set[str] = set()

    def walk(value) -> None:
        if isinstance(value, dict):
            if isinstance(value.get("blob"), str) and isinstance(
                value.get("bytes"), int
            ):
                found.add(value["blob"])
            for item in value.values():
                walk(item)
        elif isinstance(value, list):
            for item in value:
                walk(item)

    for entry in entries:
        walk(entry)
    return found


@dataclass(frozen=True)
class BlockUsed:
    seq: int
    block: int
    hash: str


def blocks_used(entries: Iterable[dict]) -> list[BlockUsed]:
    """Every block hash an entry says it decided on: seeds of rounds and of check draws."""
    used = []
    for entry in entries:
        if entry.get("kind") in ("settle", "reveal"):
            given = entry.get("input") or {}
            if (
                given.get("block_hash") is not None
                and given.get("seed_block") is not None
            ):
                used.append(
                    BlockUsed(
                        entry["seq"], int(given["seed_block"]), str(given["block_hash"])
                    )
                )
    return used


def request_body(
    request: dict, blobs: Callable[[str], Optional[bytes]]
) -> Optional[bytes]:
    body = request.get("body")
    if isinstance(body, str):
        return body.encode()
    if isinstance(body, dict) and isinstance(body.get("blob"), str):
        found = blobs(body["blob"])
        if found is not None and sha256(found) == body["blob"]:
            return found
    return None


def request_signed(request: dict, blobs: Callable[[str], Optional[bytes]]) -> bool:
    """Whether the hotkey a logged request names really signed it, so the API did not make the request up."""
    body = request_body(request, blobs)
    if body is None:
        return False
    try:
        from bittensor.wallets import Keypair

        payload = signing_payload(
            request["method"],
            request["path"],
            body,
            request["timestamp"],
            request["nonce"],
        )
        return Keypair(ss58_address=request["hotkey"]).verify(
            payload, bytes.fromhex(request["signature"])
        ) and request.get("request") == request_hash(
            request["method"],
            request["path"],
            body,
            request["timestamp"],
            request["nonce"],
            request["hotkey"],
            request["signature"],
        )
    except Exception:
        return False


@dataclass(frozen=True)
class Anchor:
    """What the API committed on chain: the hash of its log at entry `seq`, included at `block`."""

    seq: int
    head: str
    block: int


def _bytes_of(value) -> Optional[bytes]:
    if isinstance(value, (bytes, bytearray)):
        return bytes(value)
    if isinstance(value, str):
        try:
            return bytes.fromhex(value.removeprefix("0x"))
        except ValueError:
            return value.encode()
    if isinstance(value, (list, tuple)) and all(isinstance(b, int) for b in value):
        return bytes(value)
    if isinstance(value, (list, tuple)) and len(value) == 1:
        return _bytes_of(value[0])
    return None


def anchor_from_commitment(record: Optional[dict]) -> Optional[Anchor]:
    """The anchor a `Commitments.CommitmentOf` record holds, as a node decodes it; None when it holds none."""
    if not record:
        return None
    fields = (record.get("info") or {}).get("fields") or []
    seq, head = None, None
    for field in fields:
        if isinstance(field, (list, tuple)) and len(field) == 1:
            field = field[0]
        if not isinstance(field, dict):
            continue
        for variant, value in field.items():
            data = _bytes_of(value)
            if data is None:
                continue
            if variant.startswith("Raw"):
                text = data.decode("utf-8", "replace")
                if text.startswith(ANCHOR_TAG) and text[len(ANCHOR_TAG) :].isdigit():
                    seq = int(text[len(ANCHOR_TAG) :])
            elif variant == "Sha256" and len(data) == 32:
                head = data.hex()
    if seq is None or head is None:
        return None
    return Anchor(seq=seq, head=head, block=int(record.get("block") or 0))


def check_anchor(
    anchor: Anchor, hash_at: Callable[[int], Optional[str]]
) -> Optional[str]:
    """A problem with the log against what the chain anchors, or None; `hash_at` gives the log's hash at an entry, None if not read yet."""
    found = hash_at(anchor.seq)
    if found is None:
        return None
    if found != anchor.head:
        return (
            f"the chain anchors {anchor.head} at entry {anchor.seq} (block {anchor.block}), "
            f"the log read has {found}: the log was rewritten or forked"
        )
    return None
