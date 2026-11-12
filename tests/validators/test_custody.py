import asyncio
import time

import aiohttp
import pytest
from aiohttp import web
from bittensor.wallets import Keypair

from desearch.attestation import Report
from desearch.manifest import UPLOAD_LOG_LATEST, log_payload, upload_log_key
from neurons.validators import custody as custody_module
from neurons.validators.custody import (
    CUSTODY_LATEST,
    Custody,
    attestation_key,
    custody_key,
    record_hash,
    record_payload,
)
from neurons.validators.governance import TIMELOCK_BLOCKS, Neuron, Snapshot
from neurons.validators.ledger import Ledger
from neurons.validators.upload_log import UploadLog
from tests.local_http import serving

H1, H2 = "a" * 64, "b" * 64
GENESIS = Keypair.create_from_uri("//custody-genesis")
NEXT = Keypair.create_from_uri("//custody-next")
ROGUE = Keypair.create_from_uri("//custody-rogue")
LEGACY = Keypair.create_from_uri("//custody-legacy")
HEAD = 1_000_000


def sign(record: dict, key: Keypair) -> dict:
    record["signature"] = key.sign(record_payload(record)).hex()
    return record


def appoint(key=GENESIS, seq=0, prev=None, version=H1, block=100, **extra) -> dict:
    record = {
        "kind": "appoint",
        "seq": seq,
        "prev": prev,
        "signer": key.ss58_address,
        "compose_hash": version,
        "attestation_sha256": "11" * 32,
        "block": block,
        "at": 1.0,
        **extra,
    }
    return sign(record, key)


def handover(
    previous: dict,
    by=GENESIS,
    to=NEXT,
    version=H2,
    block=200,
    upload_log_seq=50,
    journal_seq=None,
) -> dict:
    record = {
        "kind": "handover",
        "seq": previous["seq"] + 1,
        "prev": record_hash(previous),
        "from": by.ss58_address,
        "signer": to.ss58_address,
        "compose_hash": version,
        "attestation_sha256": "22" * 32,
        "state_sha256": "33" * 32,
        "upload_log_seq": upload_log_seq,
        "block": block,
        "at": 2.0,
    }
    if journal_seq is not None:
        record["journal_seq"] = journal_seq
    return sign(record, by)


def chain(*texts: str, age=TIMELOCK_BLOCKS + 1):
    """Validators v1..v3 (all the stake) committing `texts` in turn, `age` blocks before the block asked."""

    async def snapshots(block):
        block = block or HEAD
        neurons = tuple(Neuron(f"v{i}", f"c{i}", True, 10) for i in range(1, 4))
        commitments = (
            {f"v{i}": (block - age, texts[(i - 1) % len(texts)]) for i in range(1, 4)}
            if texts
            else {}
        )
        return Snapshot(block, neurons, "owner", "owner-hot", commitments)

    return snapshots


def attests(ok=True):
    seen = []

    async def attest(found, **expected):
        seen.append(expected)
        report = Report()
        report.add("all", ok and found.get("signer") == expected["signer"], "fake")
        return report

    attest.seen = seen
    return attest


class Bucket:
    def __init__(self, *records):
        self.files = {}
        for record in records:
            self.put(record)

    def put(self, record: dict) -> None:
        self.files[custody_key(record["seq"])] = record
        self.files[CUSTODY_LATEST] = {"seq": record["seq"]}
        self.files[attestation_key(record["signer"])] = {"signer": record["signer"]}

    async def handle(self, request):
        found = self.files.get(request.match_info["path"])
        return (
            web.json_response(found) if found is not None else web.Response(status=404)
        )


def follow(bucket: Bucket, tmp_path, snapshots, attest=None, fallback="", steps=None):
    """Polls the bucket once, or runs `steps(custody)` between polls; returns the custody."""

    async def scenario():
        async with serving(bucket.handle) as base, aiohttp.ClientSession() as http:
            custody = Custody(
                http,
                base,
                str(tmp_path / "custody.json"),
                snapshots,
                fallback=fallback,
                attest=attest or attests(),
            )
            await custody.poll()
            if steps:
                await steps(custody)
            return custody

    return asyncio.run(scenario())


@pytest.fixture(autouse=True)
def endorse_every_poll(monkeypatch):
    monkeypatch.setattr(custody_module, "ENDORSE_EVERY_S", 0)


def test_an_appointed_cvm_starts_the_chain_and_signs_everything(tmp_path):
    genesis = appoint()
    custody = follow(
        Bucket(genesis), tmp_path, chain(f"sn22tee:{H1}@{GENESIS.ss58_address}")
    )
    assert custody.current == GENESIS.ss58_address
    assert custody.for_log(7) == [GENESIS.ss58_address]
    assert custody.for_manifest({"frozen_block": 5}) == [GENESIS.ss58_address]


def test_a_cvm_validators_did_not_appoint_is_not_followed(tmp_path):
    custody = follow(
        Bucket(appoint()), tmp_path, chain(f"sn22tee:{H1}@{ROGUE.ss58_address}")
    )
    assert custody.records == [] and custody.current is None
    assert custody.for_log(7) == []


def test_an_appointment_still_inside_the_timelock_waits(tmp_path):
    custody = follow(
        Bucket(appoint()),
        tmp_path,
        chain(f"sn22tee:{H1}@{GENESIS.ss58_address}", age=10),
    )
    assert custody.records == []


def test_a_cvm_whose_attestation_fails_is_not_followed(tmp_path):
    custody = follow(
        Bucket(appoint()),
        tmp_path,
        chain(f"sn22tee:{H1}@{GENESIS.ss58_address}"),
        attest=attests(ok=False),
    )
    assert custody.records == []


def test_the_attestation_is_checked_against_the_records_version_and_document(tmp_path):
    attest = attests()
    follow(
        Bucket(appoint()),
        tmp_path,
        chain(f"sn22tee:{H1}@{GENESIS.ss58_address}"),
        attest=attest,
    )
    assert attest.seen[0]["signer"] == GENESIS.ss58_address
    assert (attest.seen[0]["compose_hash"], attest.seen[0]["document_hash"]) == (
        H1,
        "11" * 32,
    )


def test_a_handover_to_an_approved_version_moves_the_signer_at_its_boundaries(tmp_path):
    genesis = appoint()
    bucket = Bucket(genesis, handover(genesis))
    custody = follow(bucket, tmp_path, chain(f"sn22tee:{H1}@{GENESIS.ss58_address}"))
    assert len(custody.records) == 1, "the successor's version is not approved yet"
    custody = follow(bucket, tmp_path, chain(f"sn22tee:{H1},{H2}"))
    assert custody.current == NEXT.ss58_address
    assert custody.for_log(50) == [GENESIS.ss58_address]
    assert custody.for_log(51) == [NEXT.ss58_address]
    assert custody.for_manifest({"frozen_block": 200}) == [
        GENESIS.ss58_address,
        NEXT.ss58_address,
    ]
    assert custody.for_manifest({"frozen_block": 201}) == [NEXT.ss58_address]


def test_a_handover_to_a_version_not_approved_is_refused(tmp_path):
    genesis = appoint()
    custody = follow(
        Bucket(genesis, handover(genesis)),
        tmp_path,
        chain(f"sn22tee:{H1}@{GENESIS.ss58_address}"),
    )
    assert custody.current == GENESIS.ss58_address and len(custody.records) == 1


@pytest.mark.parametrize(
    "tamper",
    [
        lambda g: handover(g, by=ROGUE),
        lambda g: {**handover(g), "upload_log_seq": 10},
        lambda g: sign({**handover(g), "prev": "00" * 32}, GENESIS),
        lambda g: sign({**handover(g), "seq": 2}, GENESIS),
        lambda g: sign({**handover(g), "kind": "takeover"}, GENESIS),
    ],
    ids=[
        "signed by another key",
        "changed after signing",
        "wrong prev",
        "seq gap",
        "unknown kind",
    ],
)
def test_a_record_that_does_not_continue_the_chain_is_refused(tmp_path, tamper):
    genesis = appoint()
    bucket = Bucket(genesis)
    bucket.files[custody_key(1)] = tamper(genesis)
    bucket.files[CUSTODY_LATEST] = {"seq": 1}
    bucket.files[attestation_key(NEXT.ss58_address)] = {"signer": NEXT.ss58_address}
    custody = follow(bucket, tmp_path, chain(f"sn22tee:{H1}@{GENESIS.ss58_address}"))
    custody = follow(bucket, tmp_path, chain(f"sn22tee:{H1},{H2}"))
    assert custody.current == GENESIS.ss58_address and len(custody.records) == 1


def test_a_second_valid_record_at_an_accepted_seq_is_a_fork_and_nobody_is_followed(
    tmp_path,
):
    genesis = appoint()
    first = handover(genesis)
    bucket = Bucket(genesis, first)
    follow(bucket, tmp_path, chain(f"sn22tee:{H1}@{GENESIS.ss58_address}"))
    custody = follow(bucket, tmp_path, chain(f"sn22tee:{H1},{H2}"))
    assert custody.current == NEXT.ss58_address
    bucket.put(handover(genesis, to=ROGUE))
    custody = follow(bucket, tmp_path, chain(f"sn22tee:{H1},{H2}"))
    assert custody.fork and [record_hash(r) for r in custody.fork][0] == record_hash(
        first
    )
    assert (
        custody.current is None
        and custody.for_log(60) == []
        and custody.for_manifest({}) == []
    )
    custody = follow(bucket, tmp_path, chain(f"sn22tee:{H1},{H2}"))
    assert custody.fork, "the fork is kept across restarts"


def test_a_replaced_record_nobody_could_have_signed_is_not_a_fork(tmp_path):
    genesis = appoint()
    bucket = Bucket(genesis)
    follow(bucket, tmp_path, chain(f"sn22tee:{H1}@{GENESIS.ss58_address}"))
    bucket.put({**appoint(), "at": 9.0})
    custody = follow(bucket, tmp_path, chain(f"sn22tee:{H1}@{GENESIS.ss58_address}"))
    assert not custody.fork and custody.current == GENESIS.ss58_address


def test_a_version_validators_stop_listing_is_no_longer_followed(tmp_path):
    genesis = appoint()
    bucket = Bucket(genesis)
    follow(bucket, tmp_path, chain(f"sn22tee:{H1}@{GENESIS.ss58_address}"))
    custody = follow(bucket, tmp_path, chain(f"sn22tee:{H2}"))
    assert custody.current is None and custody.for_log(1) == []
    custody = follow(bucket, tmp_path, chain(f"sn22tee:{H1},{H2}", age=1))
    assert (
        custody.current == GENESIS.ss58_address
    ), "endorsed again, with no timelock for the running version"


def test_before_any_record_the_fallback_signs_and_hands_over_at_the_genesis_boundary(
    tmp_path,
):
    bucket = Bucket()
    custody = follow(bucket, tmp_path, chain(), fallback=LEGACY.ss58_address)
    assert (
        custody.for_log(3) == [LEGACY.ss58_address]
        and str(custody) == LEGACY.ss58_address
    )
    bucket.put(appoint(upload_log_seq=40))
    custody = follow(
        bucket,
        tmp_path,
        chain(f"sn22tee:{H1}@{GENESIS.ss58_address}"),
        fallback=LEGACY.ss58_address,
    )
    assert custody.for_log(40) == [LEGACY.ss58_address]
    assert custody.for_log(41) == [GENESIS.ss58_address]
    assert custody.for_manifest({"frozen_block": 100}) == [
        LEGACY.ss58_address,
        GENESIS.ss58_address,
    ]


def test_the_first_record_may_be_the_legacy_signers_handover_to_an_approved_cvm(
    tmp_path,
):
    genesis = {
        "kind": "handover",
        "seq": 0,
        "prev": None,
        "from": LEGACY.ss58_address,
        "signer": GENESIS.ss58_address,
        "compose_hash": H1,
        "attestation_sha256": "22" * 32,
        "state_sha256": "33" * 32,
        "upload_log_seq": 40,
        "block": 200,
        "at": 2.0,
    }
    approved = chain(f"sn22tee:{H1}")
    for name in "bc":
        (tmp_path / name).mkdir()
    custody = follow(
        Bucket(sign(dict(genesis), LEGACY)),
        tmp_path,
        approved,
        fallback=LEGACY.ss58_address,
    )
    assert custody.current == GENESIS.ss58_address
    assert custody.for_log(40) == [LEGACY.ss58_address]
    assert custody.for_log(41) == [GENESIS.ss58_address]
    forged = follow(
        Bucket(sign(dict(genesis, **{"from": ROGUE.ss58_address}), ROGUE)),
        tmp_path / "b",
        approved,
        fallback=LEGACY.ss58_address,
    )
    assert not forged.records, "only the legacy signer may hand over first"
    unpinned = follow(Bucket(sign(dict(genesis), LEGACY)), tmp_path / "c", approved)
    assert not unpinned.records, "nobody may hand over first without a legacy signer"


def test_a_genesis_without_a_boundary_ends_the_fallback_at_once(tmp_path):
    custody = follow(
        Bucket(appoint()),
        tmp_path,
        chain(f"sn22tee:{H1}@{GENESIS.ss58_address}"),
        fallback=LEGACY.ss58_address,
    )
    assert custody.for_log(0) == [GENESIS.ss58_address]


def test_the_upload_log_takes_each_file_from_the_signer_of_its_turn(tmp_path):
    genesis = appoint()
    custody_bucket = Bucket(genesis, handover(genesis, upload_log_seq=1))

    def log_file(seq, key):
        body = {
            "seq": seq,
            "written_at": time.time(),
            "entries": [
                {
                    "task_id": f"t{seq}",
                    "miner": "m",
                    "key": f"t{seq}",
                    "completed_at": time.time(),
                    "assigned": 1,
                    "rows": 1,
                    "ok": 1,
                    "errors": 0,
                }
            ],
            "signer": key.ss58_address,
        }
        body["signature"] = key.sign(log_payload(body)).hex()
        return body

    files = {
        **custody_bucket.files,
        upload_log_key(1): log_file(1, GENESIS),
        upload_log_key(2): log_file(2, NEXT),
        upload_log_key(3): log_file(3, GENESIS),
        UPLOAD_LOG_LATEST: {"seq": 3},
    }

    async def handle(request):
        found = files.get(request.match_info["path"])
        return (
            web.json_response(found) if found is not None else web.Response(status=404)
        )

    async def scenario():
        async with serving(handle) as base, aiohttp.ClientSession() as http:
            custody = Custody(
                http,
                base,
                str(tmp_path / "custody.json"),
                chain(f"sn22tee:{H1}@{GENESIS.ss58_address}"),
                attest=attests(),
            )
            await custody.poll()
            custody.snapshots = chain(f"sn22tee:{H1},{H2}")
            await custody.poll()
            reader = UploadLog(http, base, custody, Ledger(":memory:"))
            return [len((await reader.read(seq))[0]["entries"]) for seq in (1, 2, 3)]

    assert asyncio.run(scenario()) == [
        1,
        1,
        0,
    ], "the old signer's file after its last seq holds nothing"


def test_the_hash_chained_log_changes_holder_at_the_records_journal_seq(tmp_path):
    genesis = appoint(journal_seq=0)
    bucket = Bucket(genesis, handover(genesis, journal_seq=812))
    follow(bucket, tmp_path, chain(f"sn22tee:{H1}@{GENESIS.ss58_address}"))
    custody = follow(bucket, tmp_path, chain(f"sn22tee:{H1},{H2}"))
    assert custody.for_journal(812) == [GENESIS.ss58_address]
    assert custody.for_journal(813) == [NEXT.ss58_address]
    assert custody.every() == [GENESIS.ss58_address, NEXT.ss58_address]
