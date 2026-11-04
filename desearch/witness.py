"""What validators witness and sign so the task API needs no trust on the network path or in storage.

A relaying validator forwards a miner's signed request to the task API unchanged and signs a witness record of
it: the request's hash, the miner's signature over it, the block it was forwarded at and what came back. A
witnessed request the API's log should hold, but does not once enough blocks have passed, is a censorship proof
anyone can check: the miner signed it, an independent validator forwarded it, the log lacks it.

Every validator also looks at every completed upload in storage and signs what it saw, so what storage held is
on record from many hands rather than the bucket owner's word.
"""

from __future__ import annotations

import base64
import hashlib
import json
from collections.abc import Awaitable, Callable

import aiohttp
from bittensor.wallets import Keypair

VERSION = 1
REQUEST = "request"
STORAGE = "storage"
CENSORSHIP = "censorship"
# A request ends in a receipt once the log has had this many blocks to show it.
CHECK_AFTER_BLOCKS = 50
WITNESS_HEADER = "X-Witness"
SIGNATURE = "witness_signature"

REQUEST_FIELDS = (
    "kind",
    "version",
    "witness",
    "request",
    "miner",
    "method",
    "path",
    "body_sha256",
    "timestamp",
    "nonce",
    "signature",
    "block",
    "forwarded_at",
    "status",
    "response_sha256",
    "promised",
    "answered_ms",
    "error",
)
STORAGE_FIELDS = (
    "kind",
    "version",
    "witness",
    "task_id",
    "miner",
    "key",
    "pinned_etag",
    "pinned_version",
    "status",
    "etag",
    "object_version",
    "size",
    "last_modified",
    "lock_mode",
    "lock_until",
    "matches",
    "block",
    "at",
    "error",
)
FIELDS = {REQUEST: REQUEST_FIELDS, STORAGE: STORAGE_FIELDS}


def canonical(value) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode()


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def signing_payload(
    method: str, path: str, body_sha256: str, timestamp: str, nonce: str
) -> bytes:
    """What a miner signs, as `desearch.client` and the task API build it."""
    return f"{method}\n{path}\n{body_sha256}\n{timestamp}\n{nonce}".encode()


def request_hash(
    method: str,
    path: str,
    body_sha256: str,
    hotkey: str,
    timestamp: str,
    nonce: str,
    signature: str,
) -> str:
    """One signed request's identity, as `task_api::auth::request_hash` gives it."""
    payload = signing_payload(method, path, body_sha256, timestamp, nonce)
    return sha256_hex(payload + f"\n{hotkey}\n{signature.lower()}".encode())


def miner_signed(record: dict) -> bool:
    """The miner named in a request witness signed the request it describes."""
    try:
        payload = signing_payload(
            record["method"],
            record["path"],
            record["body_sha256"],
            record["timestamp"],
            record["nonce"],
        )
        return Keypair(ss58_address=record["miner"]).verify(
            payload, bytes.fromhex(record["signature"])
        )
    except Exception:
        return False


def payload(record: dict) -> bytes:
    fields = FIELDS.get(record.get("kind"), ())
    return canonical({name: record.get(name) for name in fields})


def sign(keypair: Keypair, record: dict) -> dict:
    fields = FIELDS[record["kind"]]
    body = {name: record.get(name) for name in fields}
    body["version"] = VERSION
    body["witness"] = keypair.ss58_address
    return {**body, SIGNATURE: keypair.sign(payload(body)).hex()}


def verify(record: dict) -> bool:
    """Signed by the validator it names; a request witness must also carry the miner's signature and its hash."""
    try:
        signed = Keypair(ss58_address=record["witness"]).verify(
            payload(record), bytes.fromhex(record[SIGNATURE])
        )
    except Exception:
        return False
    if not signed:
        return False
    if record.get("kind") != REQUEST:
        return record.get("kind") == STORAGE
    hashed = request_hash(
        record["method"],
        record["path"],
        record["body_sha256"],
        record["miner"],
        record["timestamp"],
        record["nonce"],
        record["signature"],
    )
    return hashed == record["request"] and miner_signed(record)


def header_value(record: dict) -> str:
    return base64.b64encode(canonical(record)).decode()


def from_header(value: str) -> dict | None:
    try:
        return json.loads(base64.b64decode(value))
    except Exception:
        return None


# Answers a proxy or gateway gives for a request it could not pass on; the API itself never sends these.
GATEWAY_STATUSES = frozenset({502, 504, *range(520, 531)})


def expects_receipt(record: dict) -> bool:
    """Whether the log owes this request a receipt.

    One that got no answer, or a gateway's answer that it never got through, must show up: the operator's side
    either dropped it or never let it in. One whose answer carried signed receipts must show up too. One the API
    answered without a receipt is not owed one today: a refusal (a bad nonce, a claim not held, a poll over its
    rate), a repeated completion answered from the first one, or a busy 503. Once every signed input is logged
    (block B), every witnessed request is owed an entry.
    """
    status = record.get("status")
    return (
        status is None
        or status in GATEWAY_STATUSES
        or (record.get("promised") or 0) > 0
    )


LogLookup = Callable[[str], Awaitable[bool | None]]


async def censorship_proof(
    record: dict,
    head_block: int,
    included: LogLookup,
    after_blocks: int = CHECK_AFTER_BLOCKS,
    source: str = "",
) -> dict | None:
    """A proof that the log lacks a witnessed request it owes an entry, or None.

    `included(request_hash)` answers True when the log holds the request, False when it does not, None when it
    can't tell yet; nothing is claimed before `after_blocks` blocks have passed since it was forwarded.
    """
    if record.get("kind") != REQUEST or not verify(record):
        return None
    if not expects_receipt(record):
        return None
    if head_block < int(record["block"]) + after_blocks:
        return None
    if await included(record["request"]) is not False:
        return None
    return {
        "kind": CENSORSHIP,
        "version": VERSION,
        "request": record["request"],
        "miner": record["miner"],
        "witness": record,
        "checked_at_block": head_block,
        "after_blocks": after_blocks,
        "log": source,
    }


def check_proof(proof: dict, after_blocks: int = CHECK_AFTER_BLOCKS) -> bool:
    """A proof's witness is genuine and old enough; whether the log lacks it is for the reader to look up."""
    record = proof.get("witness") or {}
    return (
        proof.get("kind") == CENSORSHIP
        and verify(record)
        and expects_receipt(record)
        and proof.get("request") == record.get("request")
        and int(proof.get("checked_at_block", 0))
        >= int(record.get("block", 0)) + after_blocks
    )


RECEIPT_FIELDS = (
    "round_id",
    "hotkey",
    "requested_at",
    "outcome",
    "seq",
    "task_id",
    "refusal",
    "cause",
    "block",
    "request",
)


def receipt_body(round_id: str, entry: dict) -> dict:
    fields = {"round_id": round_id, **entry}
    return {
        name: fields[name] for name in RECEIPT_FIELDS if fields.get(name) is not None
    }


class ApiLog:
    """The log as the task API keeps it until block B anchors it on chain: receipts signed by the API's key.

    A request counts as included only through a receipt that names it and verifies against `signer`, which the
    reader pins rather than takes from the API.
    """

    def __init__(self, http, api_url: str, signer: str, timeout: float = 30.0):
        self.http = http
        self.api_url = api_url.rstrip("/")
        self.signer = Keypair(ss58_address=signer)
        self.timeout = timeout

    async def get(self, path: str) -> dict | None:
        try:
            async with self.http.get(
                f"{self.api_url}{path}",
                timeout=aiohttp.ClientTimeout(total=self.timeout),
            ) as response:
                if response.status != 200:
                    return None
                return await response.json(content_type=None)
        except Exception:
            return None

    async def included(self, request: str) -> bool | None:
        found = await self.get(f"/v1/events?request={request}")
        if found is None:
            return None
        rounds = {
            e.get("round_id") for e in found.get("events", []) if e.get("round_id")
        }
        if not found.get("events"):
            return False
        for round_id in rounds:
            log = await self.get(f"/v1/rounds/{round_id}/log")
            if log is None:
                return None
            for entry in log.get("entries", []):
                if entry.get("request") != request:
                    continue
                try:
                    if self.signer.verify(
                        canonical(receipt_body(round_id, entry)),
                        bytes.fromhex(entry.get("receipt_sig", "")),
                    ):
                        return True
                except Exception:
                    continue
        # Listed, but in no log signed by the pinned key: the listing alone proves nothing either way.
        return None
