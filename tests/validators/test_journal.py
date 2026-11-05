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


async def test_the_chained_log_answers_a_witness_check_once_anchored_past_the_request(
    tmp_path,
):
    import aiohttp

    from neurons.validators.journal import JournalLogView

    class Head:
        block = 200

    class Receipts:
        def __init__(self):
            self.asked = []

        async def included(self, request):
            self.asked.append(request)
            return None

    bucket = Bucket()
    first = chained(2)
    requests_entry = {
        "v": 1,
        "seq": 3,
        "prev": hash_of(first[-1]),
        "at": 1.0,
        "kind": "requests",
        "input": {"count": 1, "requests": [["a" * 64, 1.0]], "block": 195},
        "output": None,
    }
    entries = first + [requests_entry]
    bucket.put(segment_key(1), segment(1, entries, GENESIS))
    bucket.put(LATEST_KEY, latest(1, entries))
    anchor = []
    async with serving(bucket.handle) as url, aiohttp.ClientSession() as http:
        follower = JournalFollower(
            http,
            url,
            API.ss58_address,
            tmp_path,
            node({101: "0xhash101"}),
            anchored(anchor),
        )
        await follower.poll()
        receipts = Receipts()
        view = JournalLogView(
            lambda: follower, Head(), after_blocks=10, fallback=receipts
        )
        assert await view.included("a" * 64) is True
        assert (
            await view.included("b" * 64) is None
        ), "not anchored yet: the receipts are asked"
        assert receipts.asked == ["b" * 64]
        anchor.append(
            {
                "block": 196,
                "info": {
                    "fields": [
                        {"Raw8": "0x" + b"sn22j1:3".hex()},
                        {"Sha256": "0x" + hash_of(requests_entry)},
                    ]
                },
            }
        )
        await follower.poll()
        assert (
            await view.included("b" * 64) is False
        ), "the anchored log lists blocks past the request"
        Head.block = 300
        assert (
            await view.included("b" * 64) is None
        ), "a request newer than the anchored log: can't tell"
        assert await JournalLogView(lambda: None, Head(), 10).included("b" * 64) is None


async def test_a_follower_reads_a_handed_over_log_under_each_holder_in_turn(tmp_path):
    import aiohttp

    successor = Keypair.create_from_uri("//task-api-successor")

    class Holders:
        """API holds entries up to 3, the successor after, as a custody chain's journal_seq says."""

        def for_journal(self, seq):
            return [API.ss58_address] if seq <= 3 else [successor.ss58_address]

        def every(self):
            return [API.ss58_address, successor.ss58_address]

    bucket = Bucket()
    first = chained(3)
    second = chained(3, start=3, prev=hash_of(first[-1]))
    bucket.put(segment_key(1), segment(1, first, GENESIS))
    bucket.put(segment_key(2), segment(2, second, hash_of(first[-1]), key=successor))
    later = {
        "segment": 2,
        "last": 6,
        "head": hash_of(second[-1]),
        "signer": successor.ss58_address,
    }
    later["signature"] = successor.sign(signed_payload(later)).hex()
    bucket.put(LATEST_KEY, later)
    async with serving(bucket.handle) as url, aiohttp.ClientSession() as http:
        follower = JournalFollower(
            http, url, Holders(), tmp_path / "ok", node({}), anchored([])
        )
        assert await follower.poll() == []
        assert follower.state("last") == 6

        bucket.put(segment_key(2), segment(2, second, hash_of(first[-1])))
        follower = JournalFollower(
            http, url, Holders(), tmp_path / "old", node({}), anchored([])
        )
        problems = await follower.poll()
        assert [p.severity for p in problems] == ["fraud"]
        assert (
            "not signed" in problems[0].what
        ), "the old holder signs nothing past its turn"


ANCHORER = Keypair.create_from_uri("//task-api-anchor-test")


def compact(n: int) -> bytes:
    return bytes([n << 2]) if n < 64 else ((n << 2) | 1).to_bytes(2, "little")


def anchor_extrinsic(seq: int, head: str, key: Keypair = ANCHORER) -> bytes:
    """A signed `Commitments.set_commitment` of `sn22j1:<seq>` and a head, as the chain keeps it in the block."""
    tag = f"sn22j1:{seq}".encode()
    fields = b"\x08" + bytes([len(tag) + 1]) + tag + b"\x83" + bytes.fromhex(head)
    body = (
        b"\x84\x00"
        + key.public_key
        + b"\x01"
        + bytes(64)
        + b"\x00\x04\x00"
        + b"\x10\x05"
        + b"\x16\x00"
        + fields
    )
    return compact(len(body)) + body


class Chain:
    """Our own node: its finalized head and the extrinsics of each block up to it."""

    def __init__(self, head: int, anchors: dict[int, list[tuple[int, str]]]):
        self.head = head
        self.blocks = {
            block: [anchor_extrinsic(seq, h) for seq, h in sent]
            for block, sent in anchors.items()
        }

    async def finalized(self) -> int:
        return self.head

    async def extrinsics(self, block: int):
        return None if block > self.head else self.blocks.get(block, [b"\x04\x00"])


def follower_on(http, url, folder, chain: Chain, **more) -> JournalFollower:
    return JournalFollower(
        http,
        url,
        API.ss58_address,
        folder,
        node({}),
        anchored([]),
        chain_head=chain.finalized,
        block_extrinsics=chain.extrinsics,
        anchor_hotkey=ANCHORER.ss58_address,
        **more,
    )


def test_an_anchor_is_read_from_the_block_its_hotkey_sent_it_in():
    from desearch.journal import anchors_in_extrinsic

    head = "ab" * 32
    sent = anchor_extrinsic(1234, head)
    assert anchors_in_extrinsic(sent, ANCHORER.public_key) == [(1234, head)]
    assert anchors_in_extrinsic(sent, MALLORY.public_key) == [], "another key's"
    assert anchors_in_extrinsic(b"\x10\x04\x00\x00\x00", ANCHORER.public_key) == []


async def test_a_follower_that_started_after_a_rollback_sees_it_in_the_anchors_on_chain(
    tmp_path,
):
    import aiohttp

    bucket = Bucket()
    old = chained(6)
    new = chained(4, prev=GENESIS, branch=True)
    # The API anchored entry 6 of its log, then restarted from a copy of entry 2 and went on from there.
    bucket.put(segment_key(1), segment(1, new, GENESIS))
    bucket.put(LATEST_KEY, latest(1, new))
    chain = Chain(200, {100: [(6, hash_of(old[-1]))], 150: [(4, hash_of(new[-1]))]})
    async with serving(bucket.handle) as url, aiohttp.ClientSession() as http:
        follower = follower_on(http, url, tmp_path, chain)
        problems = await follower.poll()
        assert [p.severity for p in problems] == ["fraud"], problems
        assert "went back from entry 6 (block 100) to 4 (block 150)" in problems[0].what
        assert follower.distrust().startswith(
            "the task API's log shows fraud at entry 6"
        )
        assert await follower.poll() == [], "a proof is written once"


async def test_a_log_file_taken_down_inside_the_buckets_retention_is_fraud(tmp_path):
    import aiohttp

    bucket = Bucket()
    first = chained(2)
    second = chained(2, start=2, prev=hash_of(first[-1]))
    third = chained(2, start=4, prev=hash_of(second[-1]))
    bucket.put(segment_key(1), segment(1, first, GENESIS))
    bucket.put(LATEST_KEY, latest(1, first))
    anchors = {1_000: [(2, hash_of(first[-1]))], 1_010: [(4, hash_of(second[-1]))]}
    async with serving(bucket.handle) as url, aiohttp.ClientSession() as http:
        chain = Chain(1_020, anchors)
        follower = follower_on(http, url, tmp_path / "recent", chain)
        assert await follower.poll() == []
        assert follower.anchored_through() == 2
        # File 2 was published, anchored at block 1010, and is gone four minutes later while file 3 is there.
        bucket.put(segment_key(3), segment(3, third, hash_of(second[-1])))
        bucket.put(LATEST_KEY, latest(3, third))
        problems = await follower.poll()
        assert [p.severity for p in problems] == ["fraud"]
        assert "took it down" in problems[0].what and problems[0].seq == 3
        assert problems[0].evidence["anchor"]["block"] == 1_010
        assert "shows fraud at entry 3" in follower.distrust()
        proof = json.loads(
            next((tmp_path / "recent" / "fraud").glob("*.json")).read_text()
        )
        assert proof["problem"]["evidence"]["written_after_block"] == 1_000

        # Read a day of blocks later, the same gap may be the bucket's own lifecycle: the follower reads on.
        bucket.put(segment_key(2), segment(2, second, hash_of(first[-1])))
        bucket.put(LATEST_KEY, latest(2, second))
        later = Chain(1_000 + 7_100, anchors)
        follower = follower_on(http, url, tmp_path / "late", later)
        assert await follower.poll() == []
        del bucket.objects[segment_key(2)]
        bucket.put(segment_key(1), segment(1, first, GENESIS))
        bucket.put(LATEST_KEY, latest(3, third))
        follower.set_state("segment", 2)
        follower.set_state("last", 2)
        follower.set_state("head", hash_of(first[-1]))
        problems = await follower.poll()
        assert [p.severity for p in problems] == ["suspect"]
        assert "older than the bucket's retention" in problems[0].what


async def test_a_gone_file_no_anchor_covers_yet_waits_for_one(tmp_path):
    import aiohttp

    bucket = Bucket()
    first = chained(2)
    second = chained(2, start=2, prev=hash_of(first[-1]))
    third = chained(2, start=4, prev=hash_of(second[-1]))
    bucket.put(segment_key(1), segment(1, first, GENESIS))
    bucket.put(LATEST_KEY, latest(1, first))
    async with serving(bucket.handle) as url, aiohttp.ClientSession() as http:
        chain = Chain(1_020, {1_000: [(2, hash_of(first[-1]))]})
        follower = follower_on(http, url, tmp_path, chain)
        assert await follower.poll() == []
        bucket.put(segment_key(3), segment(3, third, hash_of(second[-1])))
        bucket.put(LATEST_KEY, latest(3, third))
        problems = await follower.poll()
        assert [p.severity for p in problems] == ["suspect"]
        assert "waiting for an anchor" in problems[0].what
        assert follower.state("segment") == 2, "it does not read past the gap yet"
        chain.head = 1_030
        chain.blocks[1_025] = [anchor_extrinsic(5, hash_of(third[0]))]
        problems = await follower.poll()
        assert [p.severity for p in problems] == ["fraud"]


async def test_a_metagraph_the_api_logged_is_checked_against_our_node_at_its_block(
    tmp_path,
):
    import aiohttp

    listed = [
        [0, "5Owner", "5OwnerCold", True, "20000000000000", "30000000000", 0.25],
        [1, "5Miner", "5MinerCold", False, "0", "0", 0.75],
    ]
    blob = json.dumps(listed, separators=(",", ":")).encode()
    entry = {
        "v": 1,
        "seq": 1,
        "prev": GENESIS,
        "at": 1.0,
        "kind": "metagraph",
        "input": {
            "netuid": 22,
            "block": 500,
            "neurons": {"blob": sha256(blob), "bytes": len(blob)},
        },
        "output": {},
    }
    ours = [
        [0, "5Owner", "5OwnerCold", True, 20000000000000, 30000000000, 16384],
        [1, "5Miner", "5MinerCold", False, 0, 0, 49151],
    ]
    asked = []

    async def metagraph_at(netuid, block):
        asked.append((netuid, block))
        return ours

    bucket = Bucket()
    bucket.put(segment_key(1), segment(1, [entry], GENESIS))
    bucket.put(LATEST_KEY, latest(1, [entry]))
    bucket.put(blob_key(sha256(blob)), blob)
    async with serving(bucket.handle) as url, aiohttp.ClientSession() as http:

        def follower(folder):
            return JournalFollower(
                http,
                url,
                API.ss58_address,
                tmp_path / folder,
                node({}),
                anchored([]),
                metagraph_at=metagraph_at,
            )

        assert await follower("same").poll() == []
        assert asked == [(22, 500)]

        # A second coldkey for the miner: one validator would count as two.
        ours[1] = [1, "5Miner", "5OwnerCold2", False, 0, 0, 49151]
        problems = await follower("other").poll()
        assert [p.severity for p in problems] == ["fraud"]
        assert "our node does not have" in problems[0].what
        assert problems[0].evidence["node"] == [ours[1]]

        async def pruned(netuid, block):
            raise RuntimeError("State discarded")

        unread = JournalFollower(
            http,
            url,
            API.ss58_address,
            tmp_path / "pruned",
            node({}),
            anchored([]),
            metagraph_at=pruned,
        )
        problems = await unread.poll()
        assert [p.severity for p in problems] == ["suspect"]


def fake_replay(folder, lines: list[dict], version: str = "0.1.0"):
    """A replay tool that answers `--version` and prints `lines` (read from a file the test rewrites) for any log."""
    import os
    import sys

    out = folder / "replay-out.jsonl"
    out.write_text("\n".join(json.dumps(line) for line in lines))
    script = folder / "task-api-replay"
    script.write_text(
        f"#!{sys.executable}\n"
        "import sys, pathlib\n"
        f"if sys.argv[1:] == ['--version']:\n    print('task-api-replay {version}'); sys.exit(0)\n"
        "pathlib.Path(sys.argv[0]).with_name('replay-args.json').write_text(__import__('json').dumps(sys.argv[1:]))\n"
        f"print(pathlib.Path({str(out)!r}).read_text())\n"
    )
    os.chmod(script, 0o755)
    return script, out


async def test_a_fraud_keeps_the_apis_inputs_out_until_the_log_holds_again(tmp_path):
    import aiohttp

    bucket = Bucket()
    first = chained(3)
    second = chained(3, start=3, prev=hash_of(first[-1]))
    bucket.put(segment_key(1), segment(1, first, GENESIS))
    bucket.put(LATEST_KEY, latest(1, first))
    tools = tmp_path / "tools"
    tools.mkdir()
    honest = {"replayed_files": 1, "last": 3, "head": "h", "fraud": 0}
    replay, out = fake_replay(tools, [{**honest, "started": True, "since": 1}])
    anchor = []
    clock = [1_000.0]
    async with serving(bucket.handle) as url, aiohttp.ClientSession() as http:
        follower = JournalFollower(
            http,
            url,
            API.ss58_address,
            tmp_path / "v",
            node({}),
            anchored(anchor),
            str(replay),
            anchor_hotkey=ANCHORER.ss58_address,
            clock=lambda: clock[0],
        )
        assert await follower.poll() == []
        assert follower.distrust() is None
        assert "--anchors" in json.loads((tools / "replay-args.json").read_text())

        # The replay finds the API's shares are not what its rules give, at entry 2.
        out.write_text(
            json.dumps(
                {"severity": "fraud", "seq": 2, "kind": "shares", "what": "the shares"}
            )
            + "\n"
            + json.dumps({**honest, "started": True, "since": 1})
        )
        problems = await follower.poll()
        assert [p.what for p in problems] == ["the shares"]
        assert (
            follower.distrust()
            == "the task API's log shows fraud at entry 2: the shares"
        )
        assert follower.fraud_spans() == [(first[1]["at"], 1_000.0)]
        assert not (tmp_path / "v" / "replay.db").exists()

        # Started again from a snapshot past the fraud, the replay holds; trust returns once an anchor covers it.
        bucket.put(segment_key(2), segment(2, second, hash_of(first[-1])))
        bucket.put(LATEST_KEY, latest(2, second))
        out.write_text(json.dumps({**honest, "last": 6, "started": True, "since": 4}))
        clock[0] = 1_060.0
        assert await follower.poll() == []
        assert "shows fraud" in follower.distrust(), "no anchor covers entry 4 yet"
        anchor.append(
            {
                "block": 70,
                "info": {
                    "fields": [
                        {"Raw8": "0x" + b"sn22j1:5".hex()},
                        {"Sha256": "0x" + hash_of(second[1])},
                    ]
                },
            }
        )
        clock[0] = 1_120.0
        assert await follower.poll() == []
        assert follower.distrust() is None
        assert follower.fraud_spans() == [(first[1]["at"], 1_120.0)]

        # A replay that stops keeping up takes the trust away again, without a fraud.
        clock[0] = 1_120.0 + 3_600
        assert "not been read and replayed lately" in follower.distrust()


async def test_without_the_replay_or_before_a_snapshot_the_log_is_not_trusted(tmp_path):
    import aiohttp

    bucket = Bucket()
    entries = chained(3)
    bucket.put(segment_key(1), segment(1, entries, GENESIS))
    bucket.put(LATEST_KEY, latest(1, entries))
    tools = tmp_path / "tools"
    tools.mkdir()
    replay, _ = fake_replay(
        tools,
        [
            {
                "replayed_files": 1,
                "last": 3,
                "head": "h",
                "fraud": 0,
                "started": False,
                "since": None,
            }
        ],
    )
    async with serving(bucket.handle) as url, aiohttp.ClientSession() as http:
        unreplayed = JournalFollower(
            http,
            url,
            API.ss58_address,
            tmp_path / "a",
            node({}),
            anchored([]),
            anchor_hotkey=ANCHORER.ss58_address,
        )
        await unreplayed.poll()
        assert "not replayed" in unreplayed.distrust()
        waiting = JournalFollower(
            http,
            url,
            API.ss58_address,
            tmp_path / "b",
            node({}),
            anchored([]),
            str(replay),
            anchor_hotkey=ANCHORER.ss58_address,
        )
        await waiting.poll()
        assert "waits for a snapshot" in waiting.distrust()


async def test_without_an_anchor_hotkey_the_skip_is_said_published_and_trusted_by_nobody(
    tmp_path, caplog
):
    import aiohttp

    bucket = Bucket()
    entries = chained(3)
    bucket.put(segment_key(1), segment(1, entries, GENESIS))
    bucket.put(LATEST_KEY, latest(1, entries))
    tools = tmp_path / "tools"
    tools.mkdir()
    replay, _ = fake_replay(
        tools,
        [
            {
                "replayed_files": 1,
                "last": 3,
                "head": "h",
                "fraud": 0,
                "started": True,
                "since": 1,
            }
        ],
    )
    async with serving(bucket.handle) as url, aiohttp.ClientSession() as http:
        follower = JournalFollower(
            http,
            url,
            API.ss58_address,
            tmp_path / "v",
            node({}),
            anchored([]),
            str(replay),
        )
        assert await follower.poll() == []
        assert "anchors are not checked" in follower.distrust(), "not a passing check"
        assert "no anchor hotkey is pinned" in caplog.text
        assert follower.status()["anchors_checked"] is False
        assert "anchors are not checked" in follower.status()["distrust"]
        caplog.clear()
        await follower.poll()
        assert (
            "no anchor hotkey" not in caplog.text
        ), "said every few minutes, not every poll"


def test_a_replay_tool_must_run_and_be_this_releases_version(tmp_path):
    from neurons.validators.journal import crate_version, replay_problem

    toml = tmp_path / "Cargo.toml"
    toml.write_text(
        '[package]\nname = "task-api"\nversion = "0.1.0"\n\n[dependencies]\nversion = "9"\n'
    )
    assert crate_version(toml) == "0.1.0"
    replay, _ = fake_replay(tmp_path, [], version="0.1.0")
    assert replay_problem(str(replay), toml) is None
    assert "does not run" in replay_problem(str(tmp_path / "missing"), toml)
    assert "no replay tool" in replay_problem("", toml)
    old, _ = fake_replay(tmp_path, [], version="0.0.9")
    assert "build it again" in replay_problem(str(old), toml)
    other = tmp_path / "other"
    other.write_text("#!/bin/sh\necho hello\n")
    other.chmod(0o755)
    assert "is not the task API's replay tool" in replay_problem(str(other), toml)
