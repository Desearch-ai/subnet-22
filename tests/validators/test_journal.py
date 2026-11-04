import json

import pytest
from aiohttp import web
from bittensor.wallets import Keypair

from desearch.journal import (
    GENESIS,
    LATEST_KEY,
    Anchor,
    Broken,
    anchor_from_commitment,
    blob_key,
    blocks_used,
    check_anchor,
    hash_of,
    request_hash,
    request_signed,
    segment_key,
    sha256,
    signed_payload,
    signing_payload,
    verify_chain,
    verify_segment,
)
from neurons.validators.journal import JournalFollower
from tests.local_http import serving

API = Keypair.create_from_uri("//task-api-journal-test")
MALLORY = Keypair.create_from_uri("//mallory")


def chained(count: int, start: int = 0, prev: str = GENESIS, **extra) -> list[dict]:
    entries = []
    for i in range(start, start + count):
        entry = {
            "v": 1,
            "seq": i + 1,
            "prev": prev,
            "at": 1_800_000_000.0 + i * 0.137,
            "kind": "settle" if i % 3 == 1 else "test",
            "input": {
                "i": i,
                "seed_block": 100 + i,
                "block_hash": f"0xhash{100 + i}",
                **extra,
            },
            "output": {"ok": i % 2 == 0},
        }
        entries.append(entry)
        prev = hash_of(entry)
    return entries


def segment(number: int, entries: list[dict], prev: str, key: Keypair = API) -> dict:
    body = {
        "segment": number,
        "first": entries[0]["seq"],
        "last": entries[-1]["seq"],
        "prev": prev,
        "head": hash_of(entries[-1]),
        "written_at": 1.5,
        "entries": entries,
        "signer": key.ss58_address,
    }
    body["signature"] = key.sign(signed_payload(body)).hex()
    return body


def test_the_api_and_a_validator_hash_the_same_bytes():
    entry = {
        "v": 1,
        "seq": 7,
        "prev": GENESIS,
        "at": 1800000000.137,
        "kind": "test",
        "input": {"text": "naïve ✓", "n": 3, "x": 0.1, "big": 1e17, "whole": 2.0},
        "output": None,
    }
    # The same vectors are asserted by the API's Rust (`journal::tests`).
    assert (
        hash_of(entry)
        == "088e36e726253dfbbde6652955384d607dafbbbe205aed45c06cb205d8a31fe8"
    )
    assert (
        request_hash(
            "POST",
            "/v1/tasks/claim",
            b'{"count":2}',
            "1800000000",
            "a" * 32,
            "5Hotkey",
            "ABCDEF",
        )
        == "ecf7fc2f64d46766666b99367b648b73b20644f096744db8db5c190f3d354712"
    )


def test_an_altered_dropped_or_forked_entry_breaks_the_chain():
    entries = chained(6)
    assert verify_chain(0, GENESIS, entries) == (6, hash_of(entries[-1]))
    altered = json.loads(json.dumps(entries))
    altered[2]["output"]["ok"] = not altered[2]["output"]["ok"]
    with pytest.raises(Broken, match="entry 4"):
        verify_chain(0, GENESIS, altered)
    with pytest.raises(Broken, match="missing"):
        verify_chain(0, GENESIS, entries[:2] + entries[3:])
    seen = hash_of(entries[2])
    fork = chained(3, start=3, prev=hash_of(chained(3, i_forked=True)[-1]))
    with pytest.raises(Broken):
        verify_chain(3, seen, fork)
    assert verify_chain(3, seen, entries[3:])[0] == 6


def test_a_file_must_be_signed_by_the_pinned_key_and_hold_together():
    entries = chained(4)
    good = segment(1, entries, GENESIS)
    assert verify_segment(good, API.ss58_address, 0, GENESIS) == (
        4,
        hash_of(entries[-1]),
    )
    with pytest.raises(Broken, match="not signed"):
        verify_segment(
            segment(1, entries, GENESIS, key=MALLORY), API.ss58_address, 0, GENESIS
        )
    tampered = json.loads(json.dumps(good))
    tampered["entries"][1]["input"]["i"] = 99
    with pytest.raises(Broken, match="not signed"):
        verify_segment(tampered, API.ss58_address, 0, GENESIS)
    resigned = segment(1, tampered["entries"], GENESIS)
    with pytest.raises(Broken):
        verify_segment(resigned, API.ss58_address, 0, GENESIS)


def test_a_logged_request_shows_its_signer():
    miner = Keypair.create_from_uri("//miner")
    body = b'{"count":2}'
    nonce, at = "c" * 32, "1800000000"
    signature = miner.sign(
        signing_payload("POST", "/v1/tasks/claim", body, at, nonce)
    ).hex()
    request = {
        "method": "POST",
        "path": "/v1/tasks/claim",
        "hotkey": miner.ss58_address,
        "timestamp": at,
        "nonce": nonce,
        "signature": signature,
        "body": body.decode(),
        "request": request_hash(
            "POST", "/v1/tasks/claim", body, at, nonce, miner.ss58_address, signature
        ),
    }
    assert request_signed(request, lambda sha: None)
    assert not request_signed({**request, "body": '{"count":3}'}, lambda sha: None)
    blob = {"blob": sha256(body), "bytes": len(body)}
    assert request_signed(
        {**request, "body": blob}, lambda sha: body if sha == blob["blob"] else None
    )
    assert not request_signed({**request, "body": blob}, lambda sha: b"other")


@pytest.mark.parametrize(
    "fields",
    [
        [{"Raw12": "0x" + b"sn22j1:12345".hex()}, {"Sha256": "0x" + "ab" * 32}],
        [
            [{"Raw12": list(b"sn22j1:12345")}],
            [{"Sha256": list(bytes.fromhex("ab" * 32))}],
        ],
    ],
)
def test_an_anchor_is_read_from_a_commitment_however_the_node_decodes_it(fields):
    record = {"block": 77, "deposit": 0, "info": {"fields": fields}}
    assert anchor_from_commitment(record) == Anchor(seq=12345, head="ab" * 32, block=77)
    assert (
        anchor_from_commitment(
            {"block": 1, "info": {"fields": [{"Raw5": "0x" + b"hello".hex()}]}}
        )
        is None
    )
    assert anchor_from_commitment(None) is None


def test_an_anchor_that_names_another_hash_is_a_rewrite():
    anchor = Anchor(seq=3, head="00" * 32, block=10)
    assert check_anchor(anchor, lambda seq: None) is None
    assert check_anchor(Anchor(3, "aa", 10), lambda seq: "aa") is None
    assert "rewritten or forked" in check_anchor(anchor, lambda seq: "aa")


def test_the_blocks_an_entry_decided_on_are_listed_for_the_node():
    used = blocks_used(chained(6))
    assert [(u.seq, u.block, u.hash) for u in used] == [
        (2, 101, "0xhash101"),
        (5, 104, "0xhash104"),
    ]


class Bucket:
    def __init__(self):
        self.objects: dict[str, bytes] = {}

    def put(self, key: str, value) -> None:
        self.objects[key] = (
            value if isinstance(value, bytes) else json.dumps(value).encode()
        )

    async def handle(self, request: web.Request) -> web.Response:
        found = self.objects.get(request.path.lstrip("/"))
        return (
            web.Response(body=found) if found is not None else web.Response(status=404)
        )


def latest(number: int, entries: list[dict]) -> dict:
    body = {
        "segment": number,
        "last": entries[-1]["seq"],
        "head": hash_of(entries[-1]),
        "signer": API.ss58_address,
    }
    body["signature"] = API.sign(signed_payload(body)).hex()
    return body


def node(blocks: dict[int, str]):
    async def block_hash(block: int):
        return blocks.get(block)

    return block_hash


def anchored(holder: list):
    """The anchor the chain holds, read as the validator reads it from its own node."""

    async def read():
        return anchor_from_commitment(holder[0]) if holder else None

    return read


async def test_a_follower_checks_files_blocks_and_anchors(tmp_path):
    import aiohttp

    bucket = Bucket()
    first = chained(3)
    requests_entry = {
        "v": 1,
        "seq": 4,
        "prev": hash_of(first[-1]),
        "at": 1.0,
        "kind": "requests",
        "input": {"count": 1, "requests": [["r" * 64, 1.0]], "block": 103},
        "output": None,
    }
    entries = first + [requests_entry]
    second = chained(3, start=4, prev=hash_of(requests_entry))
    bucket.put(segment_key(1), segment(1, entries, GENESIS))
    bucket.put(segment_key(2), segment(2, second, hash_of(entries[-1])))
    bucket.put(LATEST_KEY, latest(2, second))
    blocks = {101: "0xhash101", 104: "0xhash104", 105: "0xhash105"}
    anchor = []
    async with serving(bucket.handle) as url, aiohttp.ClientSession() as http:
        follower = JournalFollower(
            http, url, API.ss58_address, tmp_path, node(blocks), anchored(anchor)
        )
        assert await follower.poll() == []
        assert follower.included("r" * 64) == (4, 103)
        assert follower.hash_at(7) == hash_of(second[-1])

        anchor.append(
            {
                "block": 50,
                "info": {
                    "fields": [
                        {"Raw8": "0x" + b"sn22j1:5".hex()},
                        {"Sha256": "0x" + hash_of(second[0])},
                    ]
                },
            }
        )
        assert await follower.poll() == []
        assert follower.anchored_through() == 5

        anchor[0] = {
            "block": 60,
            "info": {
                "fields": [
                    {"Raw8": "0x" + b"sn22j1:6".hex()},
                    {"Sha256": "0x" + "ee" * 32},
                ]
            },
        }
        problems = await follower.poll()
        assert [p.severity for p in problems] == ["fraud"]
        assert "rewritten or forked" in problems[0].what
        proofs = list((tmp_path / "fraud").glob("*.json"))
        assert (
            len(proofs) == 1 and json.loads(proofs[0].read_text())["anchor"]["seq"] == 6
        )


async def test_a_follower_catches_a_block_hash_its_node_does_not_have_and_a_tampered_file(
    tmp_path,
):
    import aiohttp

    bucket = Bucket()
    entries = chained(3)
    bucket.put(segment_key(1), segment(1, entries, GENESIS))
    bucket.put(LATEST_KEY, latest(1, entries))
    async with serving(bucket.handle) as url, aiohttp.ClientSession() as http:
        follower = JournalFollower(
            http,
            url,
            API.ss58_address,
            tmp_path / "a",
            node({101: "0xother"}),
            anchored([]),
        )
        problems = await follower.poll()
        assert [p.severity for p in problems] == [
            "fraud"
        ] and "our node has 0xother" in problems[0].what

        tampered = segment(1, entries, GENESIS)
        tampered["entries"][0]["output"]["ok"] = False
        bucket.put(segment_key(1), tampered)
        follower = JournalFollower(
            http,
            url,
            API.ss58_address,
            tmp_path / "b",
            node({101: "0xhash101"}),
            anchored([]),
        )
        problems = await follower.poll()
        assert [p.severity for p in problems] == ["fraud"] and "not signed" in problems[
            0
        ].what
        assert (
            await follower.poll() == []
        ), "a broken log is reported once and not read on"

        follower = JournalFollower(
            http, url, MALLORY.ss58_address, tmp_path / "c", node({}), anchored([])
        )
        problems = await follower.poll()
        assert (
            "not signed by" in problems[0].what
        ), "files signed by another key than the pinned one are not read"


async def test_a_follower_back_after_files_expired_reads_on_from_a_recent_one(tmp_path):
    import aiohttp

    bucket = Bucket()
    first = chained(2)
    second = chained(2, start=2, prev=hash_of(first[-1]))
    third = chained(2, start=4, prev=hash_of(second[-1]))
    bucket.put(segment_key(1), segment(1, first, GENESIS))
    bucket.put(LATEST_KEY, latest(1, first))
    async with serving(bucket.handle) as url, aiohttp.ClientSession() as http:
        follower = JournalFollower(
            http, url, API.ss58_address, tmp_path, node({}), anchored([])
        )
        assert await follower.poll() == []
        bucket.put(segment_key(3), segment(3, third, hash_of(second[-1])))
        bucket.put(LATEST_KEY, latest(3, third))
        problems = await follower.poll()
        assert [p.severity for p in problems] == ["suspect"] and "is gone" in problems[
            0
        ].what
        assert await follower.poll() == []
        assert follower.state("last") == 6 and follower.hash_at(6) == hash_of(third[-1])
