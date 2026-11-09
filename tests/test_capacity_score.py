import hashlib

import aiohttp
import numpy as np
import pytest
from aiohttp import web

from desearch import capacity
from desearch.capacity_chain import Commitment
from desearch.embedding import encode_vector
from neurons.validators import capacity_score as score
from neurons.validators.capacity_score import (
    Row,
    Scorer,
    commitment_row,
    open_state,
    share_report,
    voided,
)

from tests.local_http import serving

DIMS = 8
BLOCK = 500
SEED = hashlib.sha256(b"seed").digest()
TEST = capacity.Test(BLOCK, SEED)
SAMPLE_HASH = hashlib.sha256(TEST.sample_block.to_bytes(8, "big")).digest()


def block_hash(block: int) -> bytes:
    if block == BLOCK:
        return SEED
    if block == TEST.sample_block:
        return SAMPLE_HASH
    return hashlib.sha256(block.to_bytes(8, "big")).digest()


def vector_of(text: str) -> np.ndarray:
    raw = hashlib.shake_256(text.encode()).digest(DIMS)
    return np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 127.5


class Reference:
    async def embed(self, texts):
        return np.array([vector_of(text) for text in texts], dtype=np.float32)


class Chain:
    def __init__(self, coldkeys: dict[str, str]):
        self.coldkeys = coldkeys
        self.commits: dict[str, list[Commitment]] = {}
        self.fail_state = False

    async def block_hash(self, block: int) -> bytes:
        return block_hash(block)

    async def registered(self, block: int) -> dict[str, str]:
        if self.fail_state:
            raise RuntimeError("State discarded")
        return self.coldkeys

    async def commitment(
        self, hotkey: str, block: int | None = None
    ) -> Commitment | None:
        found = [
            c for c in self.commits.get(hotkey, []) if block is None or c.block <= block
        ]
        return found[-1] if found else None


class Miner:
    """A miner's work: its count, and how much of what it committed is honest."""

    def __init__(self, hotkey: str, count: int, honest=lambda index: True):
        self.hotkey, self.count = hotkey, count
        self.raws = [
            encode_vector(
                vector_of(
                    capacity.text_at(SEED, hotkey, i) if honest(i) else f"garbage {i}"
                )
            )
            for i in range(count)
        ]
        self.levels = capacity.tree_levels(
            [capacity.leaf_hash(i, raw) for i, raw in enumerate(self.raws)]
        )
        self.tree_root = self.levels[-1][0]
        self.root = capacity.commit_root(SEED, hotkey, count, self.tree_root)

    def commitment(self, block: int = BLOCK + 4) -> Commitment:
        return Commitment(block, capacity.make_tag(TEST, self.count), self.root)

    def reveal(self) -> bytes:
        picked = capacity.sample_indices(SAMPLE_HASH, self.hotkey, self.count)
        items = {i: (self.raws[i], capacity.tree_proof(self.levels, i)) for i in picked}
        return capacity.encode_reveal(self.tree_root, self.count, items)


def scorer_with(
    miners: list[Miner], reference=Reference(), reveals=None, coldkeys=None
):
    chain = Chain(coldkeys or {m.hotkey: f"cold-{m.hotkey}" for m in miners})
    for miner in miners:
        chain.commits[miner.hotkey] = [miner.commitment()]
    stored = (
        {capacity.reveal_key(BLOCK, m.hotkey): m.reveal() for m in miners}
        if reveals is None
        else reveals
    )

    async def read(key):
        return stored.get(key)

    db = open_state(":memory:")
    db.execute("INSERT INTO tests VALUES (?, ?, 'found', 0)", (BLOCK, SEED))
    return Scorer(chain, read, reference, db, DIMS), chain


async def scored(scorer: Scorer) -> dict[str, tuple]:
    await scorer.step(TEST.reveal_by + score.REVEAL_MARGIN_BLOCKS)
    rows = scorer.db.execute(
        "SELECT hotkey, chars, verdict, why FROM results"
    ).fetchall()
    return {row[0]: row[1:] for row in rows}


def state_of(scorer: Scorer) -> str:
    return scorer.db.execute("SELECT state FROM tests").fetchone()[0]


async def test_an_honest_miner_earns_all_its_characters():
    miner = Miner("5A", 120)
    scorer, _ = scorer_with([miner])
    results = await scored(scorer)
    assert results["5A"] == (capacity.chars_of(SEED, "5A", 120), score.PASS, "")
    assert state_of(scorer) == "done"


async def test_a_mostly_fake_miner_earns_nothing():
    scorer, _ = scorer_with([Miner("5A", 400, honest=lambda i: i % 2 == 0)])
    assert (await scored(scorer))["5A"] == (0, score.FAIL, "mismatch")


async def test_without_a_model_nothing_is_credited():
    scorer, _ = scorer_with([Miner("5A", 120)], reference=None)
    assert (await scored(scorer))["5A"] == (0, score.UNVERIFIED, "")


async def test_an_overwrite_after_the_samples_are_public_is_not_seen():
    honest = Miner("5A", 100)
    inflated = Miner("5A", 1000)
    reveals = {capacity.reveal_key(BLOCK, "5A"): inflated.reveal()}
    scorer, chain = scorer_with([honest], reveals=reveals)
    chain.commits["5A"].append(inflated.commitment(block=TEST.sample_block + 1))
    assert (await scored(scorer))["5A"] == (0, score.FAIL, "wrong_root")


async def test_one_missing_reveal_earns_nothing_but_keeps_the_test():
    miners = [Miner(f"5M{i}", 60) for i in range(5)]
    reveals = {capacity.reveal_key(BLOCK, m.hotkey): m.reveal() for m in miners[1:]}
    scorer, _ = scorer_with(miners, reveals=reveals)
    results = await scored(scorer)
    assert results["5M0"][1] == score.NO_REVEAL and results["5M1"][1] == score.PASS
    assert state_of(scorer) == "done"


async def test_many_missing_reveals_void_the_test_for_everyone():
    miners = [Miner(f"5M{i}", 60) for i in range(5)]
    reveals = {capacity.reveal_key(BLOCK, m.hotkey): m.reveal() for m in miners[2:]}
    scorer, _ = scorer_with(miners, reveals=reveals)
    await scored(scorer)
    assert state_of(scorer) == score.VOID
    assert scorer.report(0) == share_report([], [])


async def test_a_failed_deadline_read_is_tried_again_until_the_state_is_gone():
    scorer, chain = scorer_with([Miner("5A", 60)])
    chain.fail_state = True
    await scorer.step(TEST.deadline + 1)
    assert state_of(scorer) == "found"
    chain.fail_state = False
    await scorer.step(TEST.deadline + 2)
    assert state_of(scorer) == "read"


async def test_a_deadline_state_the_node_no_longer_has_skips_the_test():
    scorer, chain = scorer_with([Miner("5A", 60)])
    chain.fail_state = True
    await scorer.step(TEST.deadline + score.STATE_KEPT_BLOCKS)
    assert state_of(scorer) == "missed"


async def test_a_reveal_longer_than_one_chunk_is_read_whole():
    body = b"r" * 264_000

    async def handler(request: web.Request) -> web.StreamResponse:
        answer = web.StreamResponse()
        await answer.prepare(request)
        for at in range(0, len(body), 8192):
            await answer.write(body[at : at + 8192])
        await answer.write_eof()
        return answer

    async with serving(handler) as url, aiohttp.ClientSession() as http:
        read = score.public_reveal_reader(url, http)
        assert await read("capacity/1/5A.reveal") == body


async def test_an_oversized_reveal_stops_being_read_past_the_limit():
    body = b"r" * (capacity.MAX_REVEAL_BYTES * 2)

    async def handler(request: web.Request) -> web.Response:
        return web.Response(body=body)

    async with serving(handler) as url, aiohttp.ClientSession() as http:
        read = score.public_reveal_reader(url, http)
        got = await read("capacity/1/5A.reveal")
    assert capacity.MAX_REVEAL_BYTES < len(got) < len(body)


async def test_a_test_nobody_committed_to_does_not_use_up_a_worst_test():
    db = open_state(":memory:")
    scorer = Scorer(Chain({}), None, None, db, DIMS)
    db.executemany(
        "INSERT INTO tests VALUES (?, ?, 'done', 0)", [(1, b"a"), (2, b"b"), (3, b"c")]
    )
    for block in (1, 2):
        for hotkey, chars in (("5A", 100), ("5B", 100)):
            scorer.save(Row(block, hotkey, f"cold-{hotkey}", 1, chars, 2, score.PASS))
    scorer.save(Row(3, "5C", "cold-5C", 0, 0, 0, score.NO_COMMIT))
    assert scorer.report(0)["tests"] == 2


def test_a_hotkey_that_changed_coldkey_keeps_the_shares_adding_up():
    rows = [
        (1, "5A", "old", 100, score.PASS, None),
        (2, "5A", "new", 100, score.PASS, None),
        (1, "5B", "other", 100, score.PASS, None),
        (2, "5B", "other", 100, score.PASS, None),
    ]
    report = share_report(rows, [1, 2])
    assert sum(c["share"] for c in report["coldkeys"]) == pytest.approx(1.0)


async def test_a_test_block_that_left_the_chain_counts_for_nobody():
    scorer, _ = scorer_with([Miner("5A", 60)])
    scorer.db.execute("UPDATE tests SET seed = ?", (b"\0" * 32,))
    await scorer.step(TEST.reveal_by + score.REVEAL_MARGIN_BLOCKS)
    assert state_of(scorer) == "left_chain"


async def test_nothing_is_judged_before_the_reveals_are_due():
    scorer, _ = scorer_with([Miner("5A", 60)])
    await scorer.step(TEST.deadline)
    assert state_of(scorer) == "read"
    rows = scorer.db.execute("SELECT verdict FROM results").fetchall()
    assert rows == [(score.NO_REVEAL,)]


@pytest.mark.parametrize(
    "commit",
    [
        None,
        Commitment(BLOCK + 3, b"not a capacity tag", b"\1" * 32),
        Commitment(
            BLOCK + 3,
            capacity.make_tag(capacity.Test(BLOCK - 300, SEED), 9),
            b"\1" * 32,
        ),
        Commitment(
            BLOCK + 3,
            capacity.make_tag(capacity.Test(BLOCK, b"\2" * 32), 9),
            b"\1" * 32,
        ),
        Commitment(BLOCK, capacity.make_tag(TEST, 9), b"\1" * 32),
        Commitment(BLOCK - 1, capacity.make_tag(TEST, 9), b"\1" * 32),
        Commitment(BLOCK + 3, capacity.make_tag(TEST, 0), b"\1" * 32),
    ],
)
def test_a_commitment_not_made_for_this_test_does_not_count(commit):
    row, root = commitment_row(TEST, "5A", "cold", commit)
    assert (row.verdict, root) == (score.NO_COMMIT, None)


def test_a_commitment_in_the_deadline_block_counts():
    commit = Commitment(TEST.deadline, capacity.make_tag(TEST, 9), b"\1" * 32)
    row, root = commitment_row(TEST, "5A", "cold", commit)
    assert (row.verdict, row.count, root) == (score.NO_REVEAL, 9, b"\1" * 32)


def test_the_share_of_a_coldkey_is_all_its_hotkeys_on_its_top_one():
    rows = [
        (1, "5A", "cold1", 700, score.PASS, 1e-6),
        (1, "5B", "cold1", 100, score.PASS, 1e-6),
        (1, "5C", "cold2", 200, score.PASS, 2e-6),
    ]
    report = share_report(rows, [1])
    first = report["coldkeys"][0]
    assert (first["coldkey"], first["top_hotkey"]) == ("cold1", "5A")
    assert first["share"] == pytest.approx(0.8)
    assert report["coldkeys"][1]["share"] == pytest.approx(0.2)
    assert report["characters"] == 1000 and report["tests"] == 1


def test_splitting_a_gpu_across_hotkeys_does_not_change_a_coldkeys_share():
    tests = [1, 2, 3]
    other = [(b, "5Z", "cold2", 200, score.PASS, None) for b in tests]
    alone = share_report(
        [(b, "5A", "cold1", 800, score.PASS, None) for b in tests] + other, tests
    )
    split = share_report(
        [(b, f"5A{i}", "cold1", 100, score.PASS, None) for b in tests for i in range(8)]
        + other,
        tests,
    )
    assert alone["coldkeys"][0]["share"] == split["coldkeys"][0]["share"]


def test_one_missed_test_is_left_out_for_any_number_of_hotkeys():
    tests = [1, 2, 3, 4]
    steady = [(b, "5Z", "cold2", 100, score.PASS, None) for b in tests]
    missed_one = [(b, "5A", "cold1", 100, score.PASS, None) for b in tests[1:]]
    report = share_report(steady + missed_one, tests)
    assert report["coldkeys"][0]["share"] == pytest.approx(0.5)
    assert report["coldkeys"][1]["share"] == pytest.approx(0.5)


def test_real_work_is_shown_beside_the_share_and_changes_nothing():
    rows = [
        (1, "5A", "cold1", 800, score.PASS, None),
        (1, "5B", "cold2", 200, score.PASS, None),
    ]
    served = {"5A": 100.0, "5B": 300.0}
    crawled = {"5A": 50.0, "5B": 50.0}
    report = share_report(rows, [1], served, crawled)
    first, second = report["coldkeys"]
    assert first["share"] == pytest.approx(0.8)
    assert first["served_share"] == pytest.approx(0.25)
    assert first["service_ratio"] == pytest.approx(0.25 / 0.8)
    assert second["service_ratio"] == pytest.approx(0.75 / 0.2)
    assert first["crawl_share"] == pytest.approx(0.5)
    assert share_report(rows, [1])["coldkeys"][0]["served_share"] is None


def test_a_median_past_the_tight_rule_is_reported_but_still_passes():
    gap = capacity.TIGHT_MEDIAN_GAP * 2
    report = share_report([(1, "5A", "cold1", 100, score.PASS, gap)], [1])
    assert report["coldkeys"][0]["share"] == 1.0
    assert report["coldkeys"][0]["hotkeys"][0]["outcomes"]["above_tight_rule"] == 1


def test_a_void_needs_more_than_a_fifth_of_coldkeys_missing():
    def rows(missing: int, passed: int) -> list[Row]:
        return [
            Row(1, f"m{i}", f"cm{i}", 1, 0, 2, score.NO_REVEAL) for i in range(missing)
        ] + [Row(1, f"p{i}", f"cp{i}", 1, 1, 2, score.PASS) for i in range(passed)]

    assert not voided(rows(1, 4))
    assert voided(rows(2, 3))
    assert not voided([Row(1, "n", "c", 0, 0, 0, score.NO_COMMIT)])


def test_one_big_operator_withholding_cannot_void_a_test():
    theirs = [Row(1, f"b{i}", "big", 1, 0, 2, score.NO_REVEAL) for i in range(38)]
    others = [Row(1, f"p{i}", f"c{i}", 1, 1, 2, score.PASS) for i in range(9)]
    assert not voided(theirs + others)


async def test_a_scan_finds_a_test_only_after_a_quiet_run(monkeypatch):
    monkeypatch.setattr(capacity, "CANDIDATE_ONE_IN", 40)
    monkeypatch.setattr(capacity, "QUIET_BLOCKS", 10)
    db = open_state(":memory:")
    scorer = Scorer(Chain({}), None, None, db, DIMS)
    await scorer.scan(100, 3000)
    found = [row[0] for row in db.execute("SELECT block FROM tests ORDER BY block")]
    flags = {n: capacity.is_candidate(block_hash(n)) for n in range(80, 3001)}
    expected = [
        n
        for n in range(100, 3001)
        if flags[n] and not any(flags[k] for k in range(n - 10, n))
    ]
    assert found == expected and found
    assert scorer.cursor() == 3000
