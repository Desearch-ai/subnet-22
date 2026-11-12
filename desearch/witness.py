"""What validators witness and sign so the task API needs no trust on the network path or in storage.

A relaying validator forwards a miner's signed request to the task API unchanged and signs a witness record of
it: the request's hash, the miner's signature over it, the block it was forwarded at and what came back, including
the receipts the API signed in its answer or the refusal it gave. The API's answer binds it to an outcome: a
witnessed request whose outcome the anchored log lacks, contradicts, or decided long after the request was
forwarded is a censorship proof anyone can check: the miner signed it, an independent validator forwarded it at a
known block, and the API's own signed answer or log shows the rest.

Every validator also looks at every completed upload in storage and probes storage the way miners use it, and
signs what it saw, so what storage held and whether it worked is on record from many hands rather than the bucket
owner's word.
"""

from __future__ import annotations

import base64
import hashlib
import json
import re
from collections.abc import Awaitable, Callable

import aiohttp
from bittensor.wallets import Keypair

# Version 2 request records keep the API's signed receipts and its refusals; version 1 records still verify.
VERSION = 2
REQUEST = "request"
STORAGE = "storage"
PROBE = "probe"
CENSORSHIP = "censorship"
# A request ends in a receipt once the log has had this many blocks to show it.
CHECK_AFTER_BLOCKS = 50
# The API decides a request it got within this many blocks of the relay forwarding it; later is a delay of its own.
LATE_BLOCKS = 10
# Of a refusal, this much of its detail is kept.
ANSWER_CHARS = 300
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
REQUEST_FIELDS_V2 = REQUEST_FIELDS + ("receipts", "answer")
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
PROBE_FIELDS = (
    "kind",
    "version",
    "witness",
    "key",
    "ok",
    "step",
    "detail",
    "took_ms",
    "block",
    "at",
)
FIELDS = {REQUEST: REQUEST_FIELDS_V2, STORAGE: STORAGE_FIELDS, PROBE: PROBE_FIELDS}


def fields_of(record: dict) -> tuple:
    if record.get("kind") == REQUEST and record.get("version", 1) < 2:
        return REQUEST_FIELDS
    return FIELDS.get(record.get("kind"), ())


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
    return canonical({name: record.get(name) for name in fields_of(record)})


def sign(keypair: Keypair, record: dict) -> dict:
    body = {name: record.get(name) for name in FIELDS[record["kind"]]}
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
        return record.get("kind") in (STORAGE, PROBE)
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


def answered_receipts(status: int | None, raw: bytes) -> list[dict]:
    """The signed receipts the API's answer carries, as it sent them; each names the request it answers."""
    if status is None or not 200 <= status < 300:
        return []
    try:
        answer = json.loads(raw)
    except ValueError:
        return []
    if not isinstance(answer, dict):
        return []
    found = answer.get("receipts") or [answer.get("receipt")]
    return [r for r in found if isinstance(r, dict) and "body" in r and "signature" in r]


def refusal(status: int | None, raw: bytes) -> dict | None:
    """What the API said when it turned a request down: its status and the start of its detail."""
    if status is None or status < 400:
        return None
    try:
        detail = json.loads(raw).get("detail")
    except (ValueError, AttributeError):
        detail = raw.decode("utf-8", "replace")
    if not isinstance(detail, str):
        detail = json.dumps(detail)
    return {"status": status, "detail": detail[:ANSWER_CHARS]}


def request_record(
    keypair: Keypair,
    signed: dict,
    block: int | None,
    forwarded_at: float,
    status: int | None,
    raw: bytes,
    answered_ms: int | None,
    error: str | None,
) -> dict:
    """A signed witness of one request sent to the API and what came back. `signed` holds the request as its sender
    signed it: `miner` (the sender's hotkey), `method`, `path`, `body_sha256`, `timestamp`, `nonce`, `signature`.
    A claim's answer also carries upload links; only its receipts are kept, so a published record hands nobody a link."""
    receipts = answered_receipts(status, raw)
    return sign(
        keypair,
        {
            "kind": REQUEST,
            "request": request_hash(
                signed["method"],
                signed["path"],
                signed["body_sha256"],
                signed["miner"],
                signed["timestamp"],
                signed["nonce"],
                signed["signature"],
            ),
            **{name: signed[name] for name in ("miner", "method", "path", "body_sha256", "timestamp", "nonce", "signature")},
            "block": block,
            "forwarded_at": forwarded_at,
            "status": status,
            "response_sha256": hashlib.sha256(raw).hexdigest() if status is not None else None,
            "promised": len(receipts),
            "answered_ms": answered_ms,
            "error": error,
            "receipts": receipts,
            "answer": refusal(status, raw),
        },
    )


def seal(keypair: Keypair, document: dict) -> dict:
    """A proof or report signed by the validator that made it, so it stands on its own wherever it is passed on."""
    body = {k: v for k, v in document.items() if k != SIGNATURE}
    body["witness"] = keypair.ss58_address
    return {**body, SIGNATURE: keypair.sign(canonical(body)).hex()}


def sealed(document: dict) -> bool:
    try:
        body = {k: v for k, v in document.items() if k != SIGNATURE}
        return Keypair(ss58_address=document["witness"]).verify(
            canonical(body), bytes.fromhex(document[SIGNATURE])
        )
    except Exception:
        return False


# Answers a proxy or gateway gives for a request it could not pass on. The API answers 502 itself when storage fails a
# completion, and logs that refusal, so a 502 is owed an entry either way.
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


def check_proof(
    proof: dict,
    after_blocks: int = CHECK_AFTER_BLOCKS,
    signers: list[str] | None = None,
    late_blocks: int = LATE_BLOCKS,
) -> bool:
    """A proof's witness is genuine and old enough, and what it quotes holds: a receipt it says the API signed was
    signed by a pinned key (`signers`), and a late entry was judged after the window. Whether the anchored log lacks
    or holds what the proof says is for the reader to look up."""
    record = proof.get("witness") or {}
    reason = proof.get("reason", MISSING)
    if not (
        proof.get("kind") == CENSORSHIP
        and verify(record)
        and proof.get("request") == record.get("request")
        and int(proof.get("checked_at_block", 0))
        >= int(record.get("block", 0)) + after_blocks
    ):
        return False
    if reason == MISSING:
        return expects_receipt(record)
    receipt = proof.get("receipt")
    if receipt is not None:
        if not any(_same_receipt(receipt, r) for r in record.get("receipts") or []):
            return False
        if signers is not None and not receipt_signed(receipt, signers):
            return False
    elif not owes_outcome(record):
        return False
    if reason == LATE:
        decided = [decided_block(e) for e in proof.get("entries") or []]
        return any(d is not None and d > int(record["block"]) + late_blocks for d in decided)
    if reason == CONTRADICTED:
        return bool(proof.get("entries"))
    return reason == UNLOGGED


# Writes whose every answer the API logs: completions and abandons (accepted, or refused with why), and validators'
# storage witnesses. A claim answered without a receipt (a rate limit, a repeated refusal) leaves no entry by design.
OWED_PATHS = re.compile(r"^/v1/tasks/[^/]+/(complete|abandon)$|^/v1/storage/witness$")
# Answers that leave no entry: a failed signature check or stale timestamp, the cap on logged refusals, and the API
# failing in itself (500, or 503 while it can't reach its own state).
UNLOGGED_STATUSES = frozenset({401, 429, 500, 503})
LATE = "late"
UNLOGGED = "unlogged"
CONTRADICTED = "contradicted"
MISSING = "missing"


def receipt_signed(receipt: dict, signers: list[str]) -> bool:
    """A receipt signed by one of the keys that held the API's log; the caller pins them, never takes them from the API."""
    for signer in signers:
        try:
            if Keypair(ss58_address=signer).verify(
                canonical(receipt["body"]), bytes.fromhex(receipt["signature"])
            ):
                return True
        except Exception:
            continue
    return False


def owes_outcome(record: dict) -> bool:
    """Whether the log owes this request an entry naming it, beyond the receipts its answer carried (checked by themselves)."""
    if record.get("version", 1) < 2 or not OWED_PATHS.match(record.get("path") or ""):
        return False
    if record.get("receipts"):
        return False
    status = record.get("status")
    return status not in UNLOGGED_STATUSES


def decided_block(entry: dict) -> int | None:
    """The block an outcome entry says its request was judged at: the entry's own, else its receipts'."""
    given = (entry.get("input") or {}).get("block")
    if isinstance(given, int):
        return given
    blocks = [
        r.get("body", {}).get("block")
        for r in (entry.get("output") or {}).get("receipts") or []
    ]
    blocks = [b for b in blocks if isinstance(b, int)]
    return max(blocks) if blocks else None


def _same_receipt(a: dict, b: dict) -> bool:
    return canonical(a.get("body")) == canonical(b.get("body"))


def _contradiction(record: dict, entry: dict) -> str | None:
    """How a logged outcome disagrees with what the API answered the relay, if it does."""
    refused = (entry.get("output") or {}).get("refused")
    status = record.get("status")
    if status is None or (status in GATEWAY_STATUSES and not refused):
        return None
    if 200 <= status < 300 and refused:
        return f"answered {status}, logged as refused with {refused.get('status')}"
    if status >= 400 and not refused:
        return f"answered {status}, logged as accepted"
    if refused and refused.get("status") != status:
        return f"answered {status}, logged as refused with {refused.get('status')}"
    return None


OutcomeLookup = Callable[[str, int], Awaitable[list[dict] | None]]


async def outcome_proof(
    record: dict,
    head_block: int,
    outcomes: OutcomeLookup,
    signers: list[str],
    after_blocks: int = CHECK_AFTER_BLOCKS,
    late_blocks: int = LATE_BLOCKS,
    source: str = "",
) -> tuple[dict | None, bool]:
    """Checks a witnessed request's outcome against the anchored log; returns (proof or None, settled).

    `outcomes(request_hash, by_block)` gives the anchored log's entries naming the request, [] when the log covers
    `by_block` and holds none, None while it can't tell. A proof is `unlogged` when a receipt the API signed, or an answer
    it owes an entry, is not in the log; `contradicted` when the log's entry for it says otherwise; `late` when the log
    decided it more than `late_blocks` after the relay forwarded it. Not settled means asked again later.
    """
    if record.get("kind") != REQUEST or not verify(record) or record.get("version", 1) < 2:
        return None, True
    if head_block < int(record["block"]) + after_blocks:
        return None, False
    by_block = int(record["block"]) + late_blocks
    receipts = [r for r in record.get("receipts") or [] if receipt_signed(r, signers)]
    for receipt in receipts:
        named = receipt["body"].get("request") or record["request"]
        entries = await outcomes(named, by_block)
        if entries is None:
            return None, False
        entry = next(
            (
                e
                for e in entries
                if any(_same_receipt(receipt, r) for r in (e.get("output") or {}).get("receipts") or [])
            ),
            None,
        )
        if entry is None:
            reason = CONTRADICTED if entries else UNLOGGED
            why = "the log holds another outcome for it" if entries else "the log holds no outcome for it"
            return _proof(record, reason, why, head_block, after_blocks, source, receipt=receipt, entries=entries), True
        decided = decided_block(entry)
        if named == record["request"] and decided is not None and decided > by_block:
            why = f"judged at block {decided}, {decided - int(record['block'])} blocks after it was forwarded"
            return _proof(record, LATE, why, head_block, after_blocks, source, receipt=receipt, entries=[entry]), True
    if not receipts and owes_outcome(record):
        entries = await outcomes(record["request"], by_block)
        if entries is None:
            return None, False
        if not entries:
            return _proof(record, UNLOGGED, "the API answered it and logged nothing", head_block, after_blocks, source), True
        for entry in entries:
            if why := _contradiction(record, entry):
                return _proof(record, CONTRADICTED, why, head_block, after_blocks, source, entries=[entry]), True
            decided = decided_block(entry)
            if decided is not None and decided > by_block:
                why = f"judged at block {decided}, {decided - int(record['block'])} blocks after it was forwarded"
                return _proof(record, LATE, why, head_block, after_blocks, source, entries=[entry]), True
    return None, True


def _proof(record, reason, why, head_block, after_blocks, source, receipt=None, entries=None) -> dict:
    proof = {
        "kind": CENSORSHIP,
        "version": VERSION,
        "reason": reason,
        "why": why,
        "request": record["request"],
        "miner": record["miner"],
        "witness": record,
        "checked_at_block": head_block,
        "after_blocks": after_blocks,
        "log": source,
    }
    if receipt is not None:
        proof["receipt"] = receipt
    if entries:
        # Each with its number in the log, so a reader checks it against the anchored chain.
        proof["entries"] = entries
    return proof


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
