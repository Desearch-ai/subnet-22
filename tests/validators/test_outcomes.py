"""Censorship proofs bound to outcomes: a relay keeps the receipts the API signed in its answer, or the refusal it gave,
and the anchored log must hold that same outcome, decided within a few blocks of the relay forwarding the request."""

import asyncio
import functools
import hashlib
import json

from bittensor.wallets import Keypair

from desearch import witness
from neurons.validators.journal import JournalFollower
from neurons.validators.witness import CensorshipCheck, ChainHead, OutcomeLog, WitnessStore
from tests.validators.test_relay import world

MINER = Keypair.create_from_uri("//outcome-miner")
RELAY = Keypair.create_from_uri("//outcome-relay")
LOG_KEY = Keypair.create_from_uri("//outcome-task-api")
FORWARDED_AT_BLOCK = 100


@functools.cache
def _signed(path: str, body: bytes, nonce: str) -> tuple:
    digest = hashlib.sha256(body).hexdigest()
    signature = MINER.sign(witness.signing_payload("POST", path, digest, "1700000000", nonce)).hex()
    return digest, signature


def signed(path: str, body: bytes = b"{}", nonce: str = "0123456789abcdef0123456789abcdef") -> dict:
    """The miner's request; signed once per path, as sr25519 signs the same bytes differently each time."""
    digest, signature = _signed(path, body, nonce)
    timestamp = "1700000000"
    return {
        "miner": MINER.ss58_address,
        "method": "POST",
        "path": path,
        "body_sha256": digest,
        "timestamp": timestamp,
        "nonce": nonce,
        "signature": signature,
    }


def receipt(request: str, outcome: str = "completed", block: int = FORWARDED_AT_BLOCK, key=LOG_KEY) -> dict:
    body = witness.receipt_body("r1", {"hotkey": MINER.ss58_address, "outcome": outcome, "seq": 3, "task_id": "t1", "block": block, "request": request})
    return {"body": body, "signature": key.sign(witness.canonical(body)).hex()}


def relayed(path: str, status: int | None, answer: dict | None = None) -> dict:
    raw = json.dumps(answer).encode() if answer is not None else b""
    return witness.request_record(RELAY, signed(path), FORWARDED_AT_BLOCK, 1700000000.5, status, raw, 40 if status else None, None if status else "TimeoutError")


def hash_of(path: str) -> str:
    request = signed(path)
    return witness.request_hash(*(request[n] for n in ("method", "path", "body_sha256", "miner", "timestamp", "nonce", "signature")))


def entry(request: str, seq: int = 7, block: int = FORWARDED_AT_BLOCK + 1, receipts=None, refused=None, kind="complete") -> dict:
    output = {"refused": refused} if refused else {"receipts": receipts or []}
    return {"seq": seq, "kind": kind, "at": 1700000001.0, "input": {"request": {"request": request}, "block": block}, "output": output}


def lookup(entries_by_request: dict, covered: bool = True):
    async def outcomes(request: str, by_block: int):
        found = entries_by_request.get(request, [])
        return found if found or covered else None

    return outcomes


def judged(record: dict, log, head: int = FORWARDED_AT_BLOCK + 50):
    return asyncio.run(witness.outcome_proof(record, head, log, [LOG_KEY.ss58_address], after_blocks=50, late_blocks=10, source="api"))


COMPLETE = "/v1/tasks/t1/complete"


def test_a_relayed_completion_keeps_the_receipt_the_api_signed_and_never_an_upload_link():
    hashed = hash_of(COMPLETE)
    signed_receipt = receipt(hashed)
    record = relayed(COMPLETE, 200, {"task_id": "t1", "status": "open_for_validation", "receipt": signed_receipt})
    assert witness.verify(record) and record["version"] == 2 and record["request"] == hashed
    assert record["receipts"] == [signed_receipt] and record["promised"] == 1
    claim = relayed("/v1/tasks/claim", 200, {"tasks": [{"upload": {"url": "https://storage/presigned"}}], "receipts": [receipt(hashed, "issued")]})
    assert "presigned" not in json.dumps(claim), "a published witness hands nobody the upload link"
    refused = relayed(COMPLETE, 409, {"detail": "you do not hold this claim"})
    assert refused["answer"] == {"status": 409, "detail": "you do not hold this claim"}


def test_a_receipt_the_api_signed_that_its_anchored_log_lacks_is_a_proof():
    hashed = hash_of(COMPLETE)
    signed_receipt = receipt(hashed)
    record = relayed(COMPLETE, 200, {"receipt": signed_receipt})
    proof, settled = judged(record, lookup({}))
    assert settled and proof["reason"] == witness.UNLOGGED
    assert witness.check_proof(proof, after_blocks=50, signers=[LOG_KEY.ss58_address])
    assert not witness.check_proof(proof, after_blocks=50, signers=[RELAY.ss58_address]), "only a pinned key's receipt counts"

    logged = entry(hashed, receipts=[signed_receipt])
    assert judged(record, lookup({hashed: [logged]})) == (None, True), "logged as answered, in time"


def test_a_log_that_holds_another_outcome_than_the_api_answered_is_a_proof():
    hashed = hash_of(COMPLETE)
    record = relayed(COMPLETE, 200, {"receipt": receipt(hashed)})
    otherwise = entry(hashed, refused={"status": 409, "detail": "you do not hold a live claim on this task"})
    proof, _ = judged(record, lookup({hashed: [otherwise]}))
    assert proof["reason"] == witness.CONTRADICTED and proof["entries"] == [otherwise]
    assert witness.check_proof(proof, after_blocks=50, signers=[LOG_KEY.ss58_address])


def test_an_outcome_the_log_decided_long_after_the_relay_forwarded_it_is_a_proof():
    hashed = hash_of(COMPLETE)
    late = receipt(hashed, block=FORWARDED_AT_BLOCK + 15)
    record = relayed(COMPLETE, 200, {"receipt": late})
    proof, _ = judged(record, lookup({hashed: [entry(hashed, block=FORWARDED_AT_BLOCK + 15, receipts=[late])]}))
    assert proof["reason"] == witness.LATE and "15 blocks after" in proof["why"]
    assert witness.check_proof(proof, after_blocks=50, signers=[LOG_KEY.ss58_address])


def test_a_refusal_the_api_answered_must_be_logged_as_that_refusal():
    hashed = hash_of(COMPLETE)
    record = relayed(COMPLETE, 409, {"detail": "you do not hold this claim"})
    proof, _ = judged(record, lookup({}))
    assert proof["reason"] == witness.UNLOGGED, "a bogus refusal kept out of the log"
    assert judged(record, lookup({hashed: [entry(hashed, refused={"status": 409, "detail": "x"})]})) == (None, True)
    proof, _ = judged(record, lookup({hashed: [entry(hashed, receipts=[receipt(hashed)])]}))
    assert proof["reason"] == witness.CONTRADICTED and "logged as accepted" in proof["why"]
    proof, _ = judged(record, lookup({hashed: [entry(hashed, refused={"status": 422, "detail": "x"})]}))
    assert proof["reason"] == witness.CONTRADICTED


def test_nothing_is_claimed_before_the_anchored_log_can_tell():
    hashed = hash_of(COMPLETE)
    record = relayed(COMPLETE, 200, {"receipt": receipt(hashed)})
    assert judged(record, lookup({}, covered=False)) == (None, False)
    assert judged(record, lookup({}), head=FORWARDED_AT_BLOCK + 49) == (None, False)


def test_answers_that_leave_no_entry_by_design_are_owed_none():
    rate_limited = relayed("/v1/tasks/claim", 200, {"tasks": [], "refusal": {"code": "RATE_LIMITED"}, "receipt": None})
    assert judged(rate_limited, lookup({})) == (None, True)
    for status in (401, 429, 500, 503):
        assert judged(relayed(COMPLETE, status, {"detail": "x"}), lookup({})) == (None, True)
    forged = relayed(COMPLETE, 200, {"receipt": receipt(hash_of(COMPLETE), key=RELAY)})
    assert judged(forged, lookup({})) == (None, True), "a receipt no pinned key signed proves nothing"


def test_a_repeated_completion_is_held_to_the_first_requests_receipt():
    first = "f" * 64
    answered = receipt(first)
    record = relayed(COMPLETE, 200, {"receipt": answered})
    assert judged(record, lookup({first: [entry(first, receipts=[answered])]})) == (None, True)
    proof, _ = judged(record, lookup({}))
    assert proof["reason"] == witness.UNLOGGED


class Follower:
    def __init__(self, entries: dict, covered: int):
        self.entries, self.covered = entries, covered

    def outcomes(self, request: str) -> list[dict]:
        return self.entries.get(request, [])

    def covered_block(self) -> int:
        return self.covered


class NoArrivals:
    async def included(self, request: str):
        return None


def test_the_check_publishes_a_signed_proof_for_each_outcome_the_log_breaks(tmp_path):
    store = WitnessStore(str(tmp_path / "witness.db"))
    hashed = hash_of(COMPLETE)
    store.add(relayed(COMPLETE, 200, {"receipt": receipt(hashed)}))
    head = ChainHead(lambda: asyncio.sleep(0, FORWARDED_AT_BLOCK + 60))
    asyncio.run(head.refresh())
    follower = Follower({}, covered=FORWARDED_AT_BLOCK + 40)
    check = CensorshipCheck(
        store,
        head,
        NoArrivals(),
        after_blocks=50,
        outcomes=OutcomeLog(lambda: follower),
        signers=lambda: [LOG_KEY.ss58_address],
        keypair=RELAY,
    )
    (proof,) = asyncio.run(check.check())
    assert proof["reason"] == witness.UNLOGGED and witness.sealed(proof)
    assert store.proofs() == [proof]
    assert asyncio.run(check.check()) == [], "each witness is judged once"
    store.close()


class Arrived:
    def __init__(self, answer):
        self.answer = answer

    async def included(self, request: str):
        return self.answer


def test_without_a_followed_log_a_request_nothing_came_back_for_is_checked_for_arrival(tmp_path):
    store = WitnessStore(str(tmp_path / "witness.db"))
    store.add(relayed(COMPLETE, None))
    head = ChainHead(lambda: asyncio.sleep(0, FORWARDED_AT_BLOCK + 60))
    asyncio.run(head.refresh())
    check = CensorshipCheck(store, head, Arrived(False), after_blocks=50, outcomes=OutcomeLog(lambda: None))
    (proof,) = asyncio.run(check.check())
    assert proof["reason"] == witness.MISSING and witness.check_proof(proof, after_blocks=50)
    store.close()


def test_the_follower_finds_an_answered_request_only_once_an_anchor_covers_it(tmp_path):
    follower = JournalFollower(None, "http://storage", LOG_KEY.ss58_address, tmp_path, None, None)
    hashed = hash_of(COMPLETE)
    logged = entry(hashed, seq=7, receipts=[receipt(hashed)])
    follower.index_outcome(logged)
    follower.set_state("anchored", 6)
    assert follower.outcomes(hashed) == []
    follower.set_state("anchored", 7)
    assert follower.outcomes(hashed) == [logged]


async def test_a_claim_through_a_relay_keeps_its_receipts_for_the_check(tmp_path):
    async with world(tmp_path) as w:
        from desearch.client import TaskApiClient
        from tests.validators.test_relay import MINER as RELAYED_MINER

        kept: list[dict] = []
        async with TaskApiClient(w.relay_urls[0], RELAYED_MINER, on_witness=kept.append) as client:
            await client.post("/v1/tasks/claim", {"kind": "crawl", "count": 1})
        (record,) = kept
        assert record["receipts"][0]["body"]["request"] == record["request"]
        assert "presigned" not in json.dumps(record)
