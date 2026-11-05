"""The upload log as a chain: each file names the hash of the one before and when it was written, so a validator tells a
withheld or forked file from one the bucket let go after its day."""

import asyncio
import hashlib
import json
import time

import aiohttp
from aiohttp import web

from desearch.manifest import UPLOAD_LOG_LATEST, log_payload, upload_log_key
from neurons.validators.ledger import Ledger
from neurons.validators.upload_log import NEXT_SEQ, UploadLog
from tests.local_http import serving
from tests.validators.test_crawl_pay import API_KEY, logged

GENESIS = "0" * 64


def signed(body: dict) -> bytes:
    body = {**body, "signer": API_KEY.ss58_address}
    body["signature"] = API_KEY.sign(log_payload(body)).hex()
    return json.dumps(body).encode()


def chain(count: int, written_at: float, step: float = 30.0) -> dict[str, bytes]:
    """Files 1..count as the task API writes them, each linked to the bytes of the one before, and latest.json."""
    files: dict[str, bytes] = {}
    prev, prev_at = GENESIS, None
    for seq in range(1, count + 1):
        at = written_at + seq * step
        raw = signed(
            {
                "seq": seq,
                "written_at": at,
                "prev": prev,
                "prev_written_at": prev_at,
                "entries": [logged(f"u{seq}", "m", at)],
            }
        )
        files[upload_log_key(seq)] = raw
        prev, prev_at = hashlib.sha256(raw).hexdigest(), at
    files[UPLOAD_LOG_LATEST] = signed({"seq": count, "head": prev, "written_at": prev_at})
    return files


async def reading(files: dict[str, bytes], polls: int, ledger: Ledger | None = None, **options):
    ledger = ledger or Ledger(":memory:")

    async def handle(request):
        found = files.get(request.match_info["path"])
        return web.Response(body=found) if found is not None else web.Response(status=404)

    async with serving(handle) as base, aiohttp.ClientSession() as http:
        reader = UploadLog(http, base, API_KEY.ss58_address, ledger, **options)
        added = [await reader.poll() for _ in range(polls)]
    return reader, added, ledger


def test_a_file_withheld_while_the_next_names_it_is_fraud_not_an_empty_file():
    files = chain(4, time.time() - 600)
    ledger = Ledger(":memory:")
    ledger.set_state(NEXT_SEQ, "1")
    del files[upload_log_key(2)]
    reader, added, _ = asyncio.run(reading(files, 2, ledger, grace_s=0.0))
    assert added == [1, 2], "file 1 first; then, judged, files 3 and 4"
    (problem,) = reader.problems
    assert problem["severity"] == "fraud" and "withheld" in problem["what"]
    assert problem["evidence"]["missing"] == [2]
    assert problem["evidence"]["next"]["seq"] == 3, "the API's own signed file names the one it withheld"
    assert ledger.state(NEXT_SEQ) == "5"


def test_a_missing_file_is_waited_for_before_it_is_judged():
    files = chain(3, time.time() - 600)
    ledger = Ledger(":memory:")
    ledger.set_state(NEXT_SEQ, "1")
    gone = files.pop(upload_log_key(2))
    reader, added, _ = asyncio.run(reading(files, 2, ledger))
    assert added == [1, 0] and reader.problems == [], "within the grace a slow store is not fraud"
    files[upload_log_key(2)] = gone
    reader, added, _ = asyncio.run(reading(files, 1, ledger))
    assert added == [2] and reader.problems == []


def test_a_file_older_than_the_bucket_keeps_files_is_gone_without_fraud():
    files = chain(3, time.time() - 3 * 86400)
    ledger = Ledger(":memory:")
    ledger.set_state(NEXT_SEQ, "1")
    del files[upload_log_key(1)]
    reader, added, _ = asyncio.run(reading(files, 2, ledger, grace_s=0.0))
    assert [p["severity"] for p in reader.problems] == ["suspect"]
    assert added == [0, 2]


def test_a_file_that_does_not_follow_the_one_read_before_it_is_a_fork():
    files = chain(3, time.time() - 600)
    ledger = Ledger(":memory:")
    ledger.set_state(NEXT_SEQ, "1")
    reader, added, _ = asyncio.run(reading(files, 1, ledger))
    assert added == [3] and reader.problems == []
    # The operator writes another file 3 and a file 4 that follows it: the chain read so far is not theirs.
    forked = chain(4, time.time() - 500)
    files[upload_log_key(4)] = forked[upload_log_key(4)]
    files[UPLOAD_LOG_LATEST] = forked[UPLOAD_LOG_LATEST]
    reader, added, _ = asyncio.run(reading(files, 1, ledger))
    assert [p["severity"] for p in reader.problems] == ["fraud"]
    assert "follows" in reader.problems[0]["what"] and reader.problems[0]["seq"] == 4


def test_a_joining_validator_checks_the_links_it_reads_back():
    files = chain(4, time.time() - 600)
    other = chain(4, time.time() - 900)
    files[upload_log_key(2)] = other[upload_log_key(2)]
    reader, added, ledger = asyncio.run(reading(files, 1))
    assert added == [4]
    # The swapped-in file follows nothing read, and the next file does not follow it.
    assert [(p["severity"], p["seq"]) for p in reader.problems] == [("fraud", 3), ("fraud", 2)]
    assert ledger.state(NEXT_SEQ) == "5"


def test_a_latest_file_not_signed_by_the_api_is_not_followed():
    files = chain(2, time.time() - 600)
    latest = json.loads(files[UPLOAD_LOG_LATEST])
    files[UPLOAD_LOG_LATEST] = json.dumps({**latest, "seq": 9}).encode()
    reader, added, _ = asyncio.run(reading(files, 1))
    assert added == [0]
