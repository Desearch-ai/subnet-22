import asyncio
import hashlib

from bittensor.wallets import Keypair

from desearch import witness
from desearch.witness import REQUEST, STORAGE

MINER = Keypair.create_from_uri("//witness-miner")
RELAY = Keypair.create_from_uri("//witness-relay")


def signed_request(path: str = "/v1/tasks/t1/complete", body: bytes = b"{}") -> dict:
    digest = hashlib.sha256(body).hexdigest()
    timestamp, nonce = "1700000000", "0123456789abcdef0123456789abcdef"
    signature = MINER.sign(
        witness.signing_payload("POST", path, digest, timestamp, nonce)
    ).hex()
    return {
        "kind": REQUEST,
        "request": witness.request_hash(
            "POST", path, digest, MINER.ss58_address, timestamp, nonce, signature
        ),
        "miner": MINER.ss58_address,
        "method": "POST",
        "path": path,
        "body_sha256": digest,
        "timestamp": timestamp,
        "nonce": nonce,
        "signature": signature,
        "block": 100,
        "forwarded_at": 1700000000.5,
        "status": None,
        "response_sha256": None,
        "promised": 0,
        "answered_ms": None,
        "error": "ClientConnectorError",
    }


def test_a_request_hashes_as_the_task_api_hashes_it():
    digest = hashlib.sha256(b"{}").hexdigest()
    hashed = witness.request_hash(
        "POST",
        "/v1/tasks/claim",
        digest,
        "5Hotkey",
        "1700000000",
        "0123456789abcdef0123456789abcdef",
        "ABcd",
    )
    # The vector task_api::auth's test pins, so both sides name a request alike.
    assert hashed == "70d28f38ac11f0c4eaac445d2662509f3f3ab2e403ea3ea7ec70aaf3ea63d19a"


def test_a_witness_verifies_only_with_both_signatures_and_its_own_hash():
    record = witness.sign(RELAY, signed_request())
    assert record["witness"] == RELAY.ss58_address and witness.verify(record)

    for name, value in [
        ("block", 99),
        ("status", 200),
        ("path", "/v1/tasks/t2/complete"),
    ]:
        assert not witness.verify(
            {**record, name: value}
        ), f"a changed {name} breaks the relay's signature"
    forged = signed_request()
    forged["signature"] = RELAY.sign(b"something else").hex()
    assert not witness.verify(
        witness.sign(RELAY, forged)
    ), "a relay can't witness a request the miner never signed"
    rehashed = {**signed_request(), "request": "0" * 64}
    assert not witness.verify(witness.sign(RELAY, rehashed))


def test_a_witness_travels_in_a_header_unchanged():
    record = witness.sign(RELAY, signed_request())
    assert witness.from_header(witness.header_value(record)) == record
    assert witness.from_header("not base64 json") is None


def test_a_storage_observation_is_signed_by_the_validator_that_looked():
    seen = witness.sign(
        RELAY,
        {
            "kind": STORAGE,
            "key": "uploads/abc.parquet",
            "status": 200,
            "etag": '"e"',
            "matches": True,
            "block": 7,
            "at": 1.0,
        },
    )
    assert witness.verify(seen)
    assert not witness.verify({**seen, "matches": False})


def included(answer):
    async def lookup(_: str):
        return answer

    return lookup


def test_a_request_the_log_lacks_long_after_it_was_relayed_is_a_censorship_proof():
    record = witness.sign(RELAY, signed_request())
    proof = asyncio.run(
        witness.censorship_proof(
            record, 150, included(False), after_blocks=50, source="api"
        )
    )
    assert proof["kind"] == witness.CENSORSHIP and proof["request"] == record["request"]
    assert witness.check_proof(proof, after_blocks=50)

    assert (
        asyncio.run(
            witness.censorship_proof(record, 149, included(False), after_blocks=50)
        )
        is None
    ), "too early"
    assert (
        asyncio.run(
            witness.censorship_proof(record, 150, included(True), after_blocks=50)
        )
        is None
    ), "it is logged"
    assert (
        asyncio.run(
            witness.censorship_proof(record, 150, included(None), after_blocks=50)
        )
        is None
    ), "can't tell"
    tampered = {**proof, "witness": {**record, "block": 140}}
    assert not witness.check_proof(tampered, after_blocks=50)


def test_only_requests_the_log_owes_an_entry_make_proofs():
    refused = witness.sign(RELAY, {**signed_request(), "status": 409, "error": None})
    assert not witness.expects_receipt(refused)
    assert (
        asyncio.run(
            witness.censorship_proof(refused, 500, included(False), after_blocks=50)
        )
        is None
    )
    answered = witness.sign(
        RELAY, {**signed_request(), "status": 200, "promised": 1, "error": None}
    )
    gateway = witness.sign(RELAY, {**signed_request(), "status": 502, "error": None})
    assert witness.expects_receipt(answered) and witness.expects_receipt(gateway)
    busy = witness.sign(RELAY, {**signed_request(), "status": 503, "error": None})
    repeated = witness.sign(
        RELAY, {**signed_request(), "status": 200, "promised": 0, "error": None}
    )
    assert not witness.expects_receipt(busy), "the API's own answer: it got the request"
    assert not witness.expects_receipt(repeated), "a completion answered from the first"
