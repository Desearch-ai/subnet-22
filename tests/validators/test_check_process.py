import asyncio
import json
import os
import subprocess
import sys
from types import SimpleNamespace

import aiohttp
import pytest
from aiohttp import web

from neurons.validators.crawl import CrawlValidator
from neurons.validators.scoring import FetchedPage
from neurons.validators.scoring_process import ScoringProcess, Unscorable
from tests.local_http import serving
from tests.synthetic import SEED, synthetic, synthetic_html, to_parquet


def check(rows, assigned, fetched, **options) -> tuple[dict, list[str]]:
    """The validator's result for an upload, and every sample page it fetched."""
    parquet = to_parquet(rows, task_id="t1", hotkey="m")
    asked: list[str] = []

    async def fetch(url: str, rendered: bool = False) -> FetchedPage:
        if not rendered:
            asked.append(url)
        return fetched[url]

    async def upload(_) -> web.Response:
        return web.Response(body=parquet)

    async def run():
        async with serving(upload) as base, aiohttp.ClientSession() as http:
            job = {
                "task_id": "t1",
                "miner": "m",
                "urls": assigned,
                "download_url": base + "x",
                "seed": SEED,
            }
            validator = CrawlValidator(
                SimpleNamespace(hotkey="v"), fetch, http, min_samples=10, **options
            )
            return await validator.score_task(job)

    return asyncio.run(run()), asked


def test_a_check_stops_early_once_a_task_can_no_longer_pass():
    rows, assigned, _ = synthetic(100)
    elsewhere = {
        url: FetchedPage(200, synthetic_html(n + 500)) for n, url in enumerate(assigned)
    }

    result, asked = check(rows, assigned, elsewhere)

    assert (result["verdict"], result["reason"]) == ("fail", "content_mismatch")
    assert result["sampled"] == len(asked) < 10
    assert result["mismatched"] > 2


def test_an_honest_task_gets_the_full_sample():
    rows, assigned, fetched = synthetic(100)

    result, asked = check(rows, assigned, fetched)

    assert (result["verdict"], result["sampled"]) == ("pass", 10)
    assert len(asked) == 10


def test_a_check_that_dies_is_tried_again_before_it_counts(monkeypatch):
    rows, assigned, fetched = synthetic(20)
    deaths = []
    real = CrawlValidator.run_check

    async def flaky(self, job, data):
        if not deaths:
            deaths.append(1)
            raise Unscorable("died")
        return await real(self, job, data)

    monkeypatch.setattr(CrawlValidator, "run_check", flaky)
    result, _ = check(rows, assigned, fetched)

    assert result["verdict"] == "pass" and deaths == [1]


@pytest.mark.parametrize(
    ("recent", "blamed"), [([True] * 10, True), ([False] * 10, None)]
)
def test_an_upload_that_crashes_a_healthy_checker_twice_is_marked(
    monkeypatch, recent, blamed
):
    rows, assigned, fetched = synthetic(20)

    async def dies(self, job, data):
        self.recent.extend(recent)
        raise Unscorable("died")

    monkeypatch.setattr(CrawlValidator, "run_check", dies)
    result, _ = check(rows, assigned, fetched)

    assert result["reason"] == "unscorable"
    assert result.get("crashed") is blamed


def test_new_or_failing_miners_go_first_then_the_oldest_upload_of_anyone():
    validator = CrawlValidator.__new__(CrawlValidator)
    validator.ledger = SimpleNamespace(trusted=lambda miner: miner != "new")
    waiting = [
        {"task_id": "a", "miner": "big", "completed_at": 30.0},
        {"task_id": "b", "miner": "big", "completed_at": 10.0},
        {"task_id": "c", "miner": "small", "completed_at": 20.0},
        {"task_id": "d", "miner": "new", "completed_at": 40.0},
    ]

    assert [m["task_id"] for m in validator.in_turn(waiting)] == ["d", "b", "c", "a"]


def test_the_parent_reads_plain_data_only_from_the_check():
    process = ScoringProcess(b"", [], SEED, 1, 0.8, 5.0)
    reader, writer = process.conn, process.child_conn
    writer.send_bytes(json.dumps(["samples", {"urls": []}]).encode())
    assert process._receive(1) == ("samples", {"urls": []})

    writer.send(("scored", {"verdict": "pass"}))
    with pytest.raises(Unscorable, match="garbled"):
        process._receive(1)
    reader.close()
    writer.close()


PROBE = """
import json, os, socket, sys
sys.path.insert(0, sys.argv[1])
from neurons.validators.confinement import confine
secret = os.path.join(sys.argv[1], "probe-secret.txt")
unconfined = confine()
found = {"unconfined": unconfined, "env": dict(os.environ)}
try:
    open(secret).read()
    found["secret"] = "read"
except OSError:
    found["secret"] = "refused"
try:
    socket.create_connection(("1.1.1.1", 80), timeout=3)
    found["network"] = "open"
except OSError:
    found["network"] = "refused"
import pyarrow, lxml.html
found["libraries"] = "loaded"
print(json.dumps(found))
"""


@pytest.mark.skipif(sys.platform != "linux", reason="the check is confined on Linux")
def test_a_confined_check_has_no_secrets_no_network_and_no_working_tree(tmp_path):
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    secret = os.path.join(root, "probe-secret.txt")
    with open(secret, "w") as handle:
        handle.write("hotkey")
    try:
        out = subprocess.run(
            [sys.executable, "-c", PROBE, root],
            capture_output=True,
            text=True,
            env={**os.environ, "SCRAPINGDOG_API_KEY": "secret"},
            timeout=60,
        )
    finally:
        os.remove(secret)
    found = json.loads(out.stdout.strip().splitlines()[-1])

    assert found["env"] == {}
    assert found["libraries"] == "loaded"
    if "files" not in found["unconfined"]:
        assert found["secret"] == "refused"
    if "network" not in found["unconfined"]:
        assert found["network"] == "refused"
