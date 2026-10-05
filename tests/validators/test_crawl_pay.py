import asyncio
import time

import aiohttp
import pytest
from aiohttp import web
from bittensor.wallets import Keypair

from desearch.credit import PENALTY_WINDOW_S
from desearch.manifest import UPLOAD_LOG_LATEST, log_payload, upload_log_key
from neurons.validators.ledger import Ledger
from neurons.validators.upload_log import NEXT_SEQ, UploadLog
from tests.local_http import serving

NOW = 1_000_000.0
API_KEY = Keypair.create_from_uri("//task-api-test")


def logged(key: str, miner: str, at: float, ok: int = 100, errors: int = 0) -> dict:
    return {
        "task_id": key,
        "miner": miner,
        "key": key,
        "completed_at": at,
        "assigned": 100,
        "rows": ok + errors,
        "ok": ok,
        "errors": errors,
    }


def ledger_with(uploads: int, miner: str = "m", start: float = NOW - 3600) -> Ledger:
    """A miner's uploads, a minute apart."""
    ledger = Ledger(":memory:")
    ledger.add_uploads([logged(f"u{n}", miner, start + 60 * n) for n in range(uploads)])
    return ledger


def checked(
    ledger: Ledger, n: int, verdict: str, credited: int = 90, content: int = 100
):
    ledger.record_check(
        f"u{n}",
        "m",
        NOW - 3600 + 60 * n,
        verdict,
        credited,
        100,
        content,
        100,
        at=NOW - 10 + n / 100,
    )


def test_unchecked_uploads_are_paid_at_the_rate_the_checks_paid():
    ledger = ledger_with(10)
    checked(ledger, 0, "pass")
    checked(ledger, 5, "pass")

    found = ledger.crawl_paid(NOW)["m"]

    assert found.rate == pytest.approx(0.9)
    assert found.rows == pytest.approx(2 * 90 + 8 * 90)
    assert (found.uploads, found.checked, found.failed) == (10, 2, 0)


def test_a_failed_check_takes_back_what_came_since_the_last_pass_and_costs_its_urls():
    ledger = ledger_with(10)
    checked(ledger, 2, "pass", credited=100)
    checked(ledger, 6, "fail", credited=0)

    found = ledger.crawl_paid(NOW)["m"]

    rate = 100 / 200
    kept = [0, 1, 7, 8, 9]
    assert found.rows == pytest.approx(100 + rate * 100 * len(kept) - 100)


def test_two_fails_among_the_last_ten_checks_take_back_the_day_before():
    ledger = ledger_with(10)
    checked(ledger, 1, "fail", credited=0)
    checked(ledger, 4, "pass", credited=100)
    checked(ledger, 8, "fail", credited=0)

    found = ledger.crawl_paid(NOW)["m"]

    assert found.failed == 2
    assert found.rows == pytest.approx(100 / 300 * 100 * 1 - 200), (
        "only the upload after the second fail is left, less both fails' URLs"
    )


def test_a_check_that_counted_fewer_pages_than_reported_is_a_fail():
    ledger = ledger_with(2)
    checked(ledger, 0, "pass", credited=40, content=40)

    found = ledger.crawl_paid(NOW)["m"]

    assert found.failed == 1 and found.rows < 0


def test_a_miner_this_validator_never_checked_earns_nothing_yet():
    found = ledger_with(5).crawl_paid(NOW)["m"]

    assert (found.rate, found.rows) == (0.0, 0.0)


def test_uploads_older_than_the_window_are_not_counted():
    ledger = ledger_with(3, start=NOW - 2 * PENALTY_WINDOW_S)
    ledger.add_uploads([logged("fresh", "m", NOW - 60)])
    ledger.record_check("fresh", "m", NOW - 60, "pass", 100, 100, 100, 100, at=NOW)

    assert ledger.crawl_paid(NOW)["m"].uploads == 1


def signed(seq: int, entries: list[dict], written_at: float, key=API_KEY) -> dict:
    body = {
        "seq": seq,
        "written_at": written_at,
        "entries": entries,
        "signer": API_KEY.ss58_address,
    }
    body["signature"] = key.sign(log_payload(body)).hex()
    return body


def test_the_upload_log_is_read_from_storage_in_order_and_unsigned_files_hold_nothing():
    files = {
        upload_log_key(1): signed(1, [logged("a", "m", time.time())], time.time()),
        upload_log_key(2): signed(
            2,
            [logged("b", "m", time.time())],
            time.time(),
            Keypair.create_from_uri("//x"),
        ),
        UPLOAD_LOG_LATEST: {"seq": 2},
    }

    async def handle(request):
        found = files.get(request.match_info["path"])
        return web.json_response(found) if found else web.Response(status=404)

    async def scenario():
        ledger = Ledger(":memory:")
        async with serving(handle) as base, aiohttp.ClientSession() as http:
            reader = UploadLog(http, base, API_KEY.ss58_address, ledger)
            first = await reader.poll()
            files[upload_log_key(3)] = signed(
                3, [logged("c", "m", time.time())], time.time()
            )
            files[UPLOAD_LOG_LATEST] = {"seq": 4}
            second = await reader.poll()
            return first, second, ledger.state(NEXT_SEQ)

    first, second, cursor = asyncio.run(scenario())
    assert first == 1, "a new validator reads back, and the unsigned file holds nothing"
    assert second == 1 and cursor == "4", "a file not there yet is waited for"
