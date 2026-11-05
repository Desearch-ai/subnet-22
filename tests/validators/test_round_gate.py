"""A validator judges coverage as the task API does: against its round's gate, 0.85 × the share of the round's pages
validators could load, outside the hosts most miners of the round could not load."""

import asyncio
import json
import sqlite3

import aiohttp
import pytest
from aiohttp import web

from desearch.credit import covers, outside_hosts, round_gate
from desearch.journal import GENESIS, LATEST_KEY, segment_key
from neurons.validators.crawl import CrawlValidator
from neurons.validators.journal import JournalFollower
from neurons.validators.ledger import Ledger
from neurons.validators.scoring import host_of, score
from tests.local_http import serving
from tests.synthetic import SEED, synthetic, synthetic_url, to_parquet
from tests.validators.test_journal import API, Bucket, anchored, latest, node, segment
from tests.validators.test_scoring import Votes

NOW = 1_000_000.0


def test_the_gate_falls_with_the_share_of_the_round_validators_could_load():
    assert round_gate(29, 0) == 0.85, "too few samples to tell"
    assert round_gate(30, 15) == pytest.approx(0.425)
    assert round_gate(40, 40) == 0.85
    assert (covers(17, 20, 0.85), covers(16, 20, 0.85)) == (True, False)
    assert covers(0, 0, 0.85), "a task wholly on neutral hosts is covered"
    hosts = {"a.example": [3, 1], "b.example": [2, 0]}
    assert outside_hosts(hosts, {"a.example": 4, "b.example": 6}, {"b.example"}) == (
        3,
        4,
    )


def _scored(returned: int, gate: float, neutral=()) -> dict:
    rows, assigned, fetched = synthetic(20)
    return score(rows[:returned], assigned, fetched, SEED, 5, 0.8, gate, neutral)


def test_coverage_is_judged_against_the_rounds_gate():
    short = _scored(10, 0.85)
    assert (short["verdict"], short["reason"]) == ("fail", "coverage")
    # Validators could load half the round: half the URLs are enough.
    lenient = _scored(10, 0.425)
    assert (lenient["verdict"], lenient["reason"]) == ("pass", "ok")
    # The verdict without coverage is kept for the ledger either way.
    assert short["content_verdict"] == lenient["content_verdict"] == "pass"
    assert short["assigned_hosts"] == {host_of(synthetic_url(n)): 1 for n in range(20)}


def test_rows_and_urls_on_neutral_hosts_are_left_out_of_coverage():
    missing = {host_of(synthetic_url(n)) for n in range(10, 20)}
    returned = {host_of(synthetic_url(n)) for n in range(10)}

    assert _scored(10, 0.85, missing)["verdict"] == "pass"
    assert _scored(10, 0.85, missing | returned)["verdict"] == "pass", "all neutral"
    assert _scored(10, 0.85, returned)["reason"] == "coverage"


def _check(ledger, key, round_id, rows, hosts=None, sampled=0, loaded=0, at=NOW):
    """A check that read `rows` of 100 URLs on a.example, all good pages."""
    ledger.record_check(
        key,
        "m",
        at - 3600,
        "pass",
        rows,
        rows,
        rows,
        100,
        at=at - 10,
        round_id=round_id,
        hosts=hosts or {"a.example": [rows, 0]},
        content_verdict="pass",
        assigned_hosts={"a.example": 100},
        sampled=sampled,
        loaded=loaded,
    )


def test_the_ledger_judges_coverage_on_the_gate_the_signed_log_settled():
    ledger = Ledger(":memory:")
    _check(ledger, "u0", "r1", 50)

    # Open round, nothing known of its loadability: the full gate, and half the URLs fail.
    assert ledger.crawl_paid(NOW)["m"].amount == -100
    # The task API closed the round: validators could load half of it.
    settled = {"r1": {"gate": 0.425, "neutral": []}}
    assert ledger.crawl_paid(NOW, rounds=settled)["m"].amount == 50


def test_our_own_samples_lower_the_gate_while_the_round_is_open():
    ledger = Ledger(":memory:")
    for n in range(3):
        _check(ledger, f"u{n}", "r1", 40, sampled=10, loaded=4)

    gate, neutral = ledger.round_view("r1")
    assert (gate, neutral) == (pytest.approx(0.34), set())
    # 40 of 100 URLs clear 0.85 × 12/30.
    assert ledger.crawl_paid(NOW)["m"].amount == 120


def test_the_settled_neutral_hosts_override_our_own():
    ledger = Ledger(":memory:")
    hosts = {"a.example": [50, 0], "owner.example": [50, 0]}
    ledger.record_check(
        "u0",
        "m",
        NOW - 3600,
        "pass",
        100,
        100,
        100,
        100,
        at=NOW - 10,
        round_id="r1",
        hosts=hosts,
        content_verdict="pass",
        assigned_hosts={"a.example": 50, "owner.example": 50},
    )
    assert ledger.crawl_paid(NOW)["m"].amount == 100
    settled = {"r1": {"gate": 0.85, "neutral": ["owner.example"]}}
    # The owner's host pays nobody; coverage is 50 of the 50 URLs outside it.
    assert ledger.crawl_paid(NOW, rounds=settled)["m"].amount == 50
    assert ledger.round_view("r1", settled["r1"]) == (0.85, {"owner.example"})


def test_a_ledger_from_before_keeps_the_verdicts_it_recorded(tmp_path):
    path = tmp_path / "verdicts.db"
    old = sqlite3.connect(path)
    old.executescript("""
        CREATE TABLE checks (key TEXT PRIMARY KEY, miner TEXT NOT NULL, completed_at REAL NOT NULL,
            scored_at REAL NOT NULL, verdict TEXT NOT NULL, credited INTEGER NOT NULL, returned INTEGER NOT NULL,
            content INTEGER NOT NULL, assigned INTEGER NOT NULL);
        INSERT INTO checks VALUES ('u0', 'm', 996400.0, 999990.0, 'pass', 60, 60, 60, 100);
        """)
    old.commit()
    old.close()

    ledger = Ledger(path)

    columns = {row[1] for row in ledger.db.execute("PRAGMA table_info(checks)")}
    assert {"content_verdict", "assigned_hosts", "sampled", "loaded"} <= columns
    # Recorded with coverage already judged: not judged again.
    assert ledger.crawl_paid(NOW)["m"].amount == 60


def test_a_check_takes_its_rounds_view_from_the_signed_log_once_it_closed():
    ledger = Ledger(":memory:")
    for n in range(3):
        _check(ledger, f"u{n}", "r1", 40, sampled=10, loaded=4)
    settled = {"r2": {"gate": 0.5, "neutral": ["junk.example"]}}
    checker = CrawlValidator(None, None, None, ledger=ledger, round_views=settled.get)

    assert checker.coverage_view({"round_id": "r1"}) == (pytest.approx(0.34), set())
    assert checker.coverage_view({"round_id": "r2"}) == (0.5, {"junk.example"})
    bare = CrawlValidator(None, None, None)
    assert bare.coverage_view({"round_id": "r1"}) == (0.85, set())


async def test_the_follower_keeps_each_closed_rounds_settled_view(tmp_path):
    view = {
        "sampled": 36,
        "loaded": 12,
        "loadability": 1 / 3,
        "gate": 0.85 / 3,
        "hosts": {"junk.example": {"uploads": 6, "mostly_errors": 6}},
        "neutral": ["junk.example"],
    }
    close = {
        "v": 1,
        "seq": 1,
        "prev": GENESIS,
        "at": 1.0,
        "kind": "close",
        "input": {"round_id": "r1"},
        "output": {"anchor_root": "00" * 32, "round": view, "settled": []},
    }
    bucket = Bucket()
    bucket.put(segment_key(1), segment(1, [close], GENESIS))
    bucket.put(LATEST_KEY, latest(1, [close]))
    async with serving(bucket.handle) as url, aiohttp.ClientSession() as http:
        follower = JournalFollower(
            http, url, API.ss58_address, tmp_path, node({}), anchored([])
        )
        assert await follower.poll() == []

    assert follower.round_view("r1") == view
    assert follower.round_view("r2") is None
    assert follower.round_views() == {"r1": view}


def test_a_check_votes_on_the_settled_gate_and_keeps_coverage_open_in_the_ledger():
    """Half a task's URLs in a round validators could load half of: the vote passes, and the ledger keeps the verdict without
    coverage, the URLs per host and the samples, to judge coverage again when weights are set.
    """
    rows, assigned, fetched = synthetic(10)
    body = to_parquet(rows[:5])

    async def upload(_) -> web.Response:
        return web.Response(body=body)

    async def fetch(url):
        return fetched[url]

    async def check():
        async with serving(upload) as base, aiohttp.ClientSession() as http:
            api, ledger = Votes(), Ledger(":memory:")
            checker = CrawlValidator(
                api,
                fetch,
                http,
                ledger=ledger,
                storage_url=base,
                round_views={"r1": {"gate": 0.425, "neutral": []}}.get,
            )
            job = {
                "task_id": "t1",
                "kind": "crawl",
                "miner": "m",
                "key": "uploads/t1.parquet",
                "urls": assigned,
                "completed_at": NOW - 60,
                "round_id": "r1",
            }
            await checker.check(checker.job_of(job, SEED))
            return api.posts, ledger

    ((path, vote),), ledger = asyncio.run(check())

    assert path == "/v1/validation/t1/score"
    assert (vote["verdict"], vote["reason"]) == ("pass", "ok")
    assert not {"content_verdict", "content_reason", "assigned_hosts"} & set(vote)
    kept = ledger.db.execute(
        "SELECT verdict, content_verdict, assigned_hosts, sampled, loaded FROM checks"
    ).fetchone()
    assert kept[:2] == ("pass", "pass")
    assert sum(json.loads(kept[2]).values()) == 10
    assert kept[3] == kept[4] > 0
    # Without the settled view our own five samples are too few to lower the gate: half the URLs fail.
    assert ledger.crawl_paid(NOW)["m"].amount == -10
    settled = {"r1": {"gate": 0.425, "neutral": []}}
    assert ledger.crawl_paid(NOW, rounds=settled)["m"].amount == 5
