"""Scores the capacity test from public data alone: the chain, miners' reveals and your own model server. It sets no weights.

    python -m neurons.validators.capacity_score --follow --state capacity.db --out scores.json \
        --model-url http://127.0.0.1:8000/v1/embeddings

Without --model-url it checks only that each reveal proves its commitment, and marks the result unverified. Commitments are read
as of each test's deadline block, which a pruned node keeps for about 256 blocks, so follow the chain or point at an archive node.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sqlite3
import statistics
import time
from collections import defaultdict
from dataclasses import dataclass, replace
from pathlib import Path

import aiohttp
import numpy as np

from desearch import capacity
from desearch.capacity_chain import CapacityChain, Commitment
from desearch.embedding import EmbeddingClient, model_named

log = logging.getLogger("capacity_score")

PUBLIC_BUCKET = "https://subnet-22.s3.us-east-va.io.cloud.ovh.us"
READS_AT_ONCE = 16
SCAN_AT_ONCE = 16
FOLLOW_S = 12.0
# Reveals may still be arriving as the last allowed block is made.
REVEAL_MARGIN_BLOCKS = 2
# A pruned node keeps state about 256 blocks back; a failed read is worth trying again until then.
STATE_KEPT_BLOCKS = 240

PASS = "pass"
FAIL = "fail"
UNVERIFIED = "unverified"
NO_COMMIT = "no_commit"
NO_REVEAL = "no_reveal"
VOID = "void"
FAILS_PROOF = ("wrong_root", "bad_proof", "malformed_vector")


@dataclass(frozen=True)
class Row:
    block: int
    hotkey: str
    coldkey: str
    count: int
    chars: int
    commit_block: int
    verdict: str
    why: str = ""
    worst: float | None = None
    median_gap: float | None = None


def open_state(path: str) -> sqlite3.Connection:
    db = sqlite3.connect(path)
    db.executescript("""
        CREATE TABLE IF NOT EXISTS cursor (id INTEGER PRIMARY KEY CHECK (id = 1), block INTEGER NOT NULL);
        CREATE TABLE IF NOT EXISTS tests (
            block INTEGER PRIMARY KEY, seed BLOB NOT NULL, state TEXT NOT NULL, at REAL NOT NULL
        );
        CREATE TABLE IF NOT EXISTS results (
            block INTEGER NOT NULL, hotkey TEXT NOT NULL, coldkey TEXT NOT NULL, count INTEGER NOT NULL,
            chars INTEGER NOT NULL, commit_block INTEGER NOT NULL, root BLOB, verdict TEXT NOT NULL, why TEXT NOT NULL,
            worst REAL, median_gap REAL, PRIMARY KEY (block, hotkey)
        );
        """)
    return db


def commitment_row(
    test: capacity.Test, hotkey: str, coldkey: str, found: Commitment | None
) -> tuple[Row, bytes | None]:
    """What a hotkey committed for this test as of its deadline, or why nothing it holds counts."""
    empty = Row(test.block, hotkey, coldkey, 0, 0, 0, NO_COMMIT)
    if found is None:
        return empty, None
    parsed = capacity.parse_tag(found.tag)
    if parsed is None:
        return empty, None
    block, prefix, count = parsed
    # A hotkey's older commitment stays on chain; only one made inside this test's blocks is for this test.
    if (
        block != test.block
        or prefix != test.seed[:4]
        or not test.block < found.block <= test.deadline
        or count == 0
    ):
        return empty, None
    return (
        Row(test.block, hotkey, coldkey, count, 0, found.block, NO_REVEAL),
        found.root,
    )


def judged(
    test: capacity.Test,
    row: Row,
    root: bytes,
    sample_hash: bytes,
    body: bytes | None,
    dims: int,
) -> tuple[Row, tuple[list[int], np.ndarray] | None]:
    """A commitment against its reveal: a missing or unproven reveal earns nothing; a proven one still needs its vectors checked."""
    if body is None:
        return row, None
    try:
        reveal = capacity.decode_reveal(body)
    except ValueError as exc:
        return replace(row, why=f"unreadable: {exc}"), None
    opened = capacity.opened_vectors(
        test.seed, row.hotkey, row.count, root, sample_hash, reveal, dims
    )
    if isinstance(opened, str):
        verdict = FAIL if opened in FAILS_PROOF else NO_REVEAL
        return replace(row, verdict=verdict, why=opened), None
    return replace(row, verdict=UNVERIFIED), opened


def with_similarity(test: capacity.Test, row: Row, values: list[float]) -> Row:
    """No verdict, no credit: characters count only after the validator's own vectors agree."""
    row = replace(row, worst=min(values), median_gap=capacity.median_gap(values))
    if not capacity.matches(values):
        return replace(row, verdict=FAIL, why="mismatch")
    chars = capacity.chars_of(test.seed, row.hotkey, row.count)
    return replace(row, verdict=PASS, chars=chars)


def voided(rows: list[Row]) -> bool:
    """Many owners' reveals missing at once means the path failed, not the miners; the test counts for nobody.

    It counts coldkeys, so one big operator withholding its own cannot void a test for everyone.
    """
    committed = {row.coldkey for row in rows if row.verdict != NO_COMMIT}
    revealed = {
        row.coldkey for row in rows if row.verdict not in (NO_COMMIT, NO_REVEAL)
    }
    return (
        bool(committed)
        and len(committed - revealed) / len(committed) > capacity.VOID_MISSING_SHARE
    )


def share_report(
    rows: list[tuple],
    tests: list[int],
    served: dict[str, float] | None = None,
    crawled: dict[str, float] | None = None,
) -> dict:
    """Each coldkey's share of the window's verified characters, credited to its top hotkey, as pay would be.

    A coldkey's worst test is left out, so one missed commitment is not a lost day. That happens per coldkey, not per hotkey, so
    it never favours more hotkeys. `served` and `crawled` are the same hotkeys' real work: a high share with little served work
    is capacity that comes up for tests alone, and crawl is still shared by hotkey, so both are shown beside the test's result.
    """
    by_hotkey: dict[str, int] = defaultdict(int)
    by_test: dict[str, dict[int, int]] = defaultdict(lambda: defaultdict(int))
    coldkeys: dict[str, set[str]] = defaultdict(set)
    outcomes: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    gaps: list[float] = []
    for block, hotkey, coldkey, chars, verdict, gap in rows:
        by_hotkey[hotkey] += chars
        by_test[coldkey][block] += chars
        coldkeys[coldkey].add(hotkey)
        outcomes[hotkey][verdict] += 1
        if gap is not None:
            gaps.append(gap)
            if gap > capacity.TIGHT_MEDIAN_GAP:
                outcomes[hotkey]["above_tight_rule"] += 1
    kept = {
        coldkey: sum(values) - (min(values) if len(values) > 1 else 0)
        for coldkey, values in (
            (coldkey, [got.get(block, 0) for block in tests])
            for coldkey, got in by_test.items()
        )
    }
    total = sum(kept.values())
    work_blocks = capacity.WINDOW_BLOCKS - capacity.REFERENCE_STOP_BLOCKS
    window_s = max(len(tests) - 1, 1) * work_blocks * capacity.SECONDS_PER_BLOCK

    def of(amounts: dict[str, float] | None, hotkeys: set[str]) -> float | None:
        if not amounts or not sum(amounts.values()):
            return None
        return sum(amounts.get(h, 0.0) for h in hotkeys) / sum(amounts.values())

    report = []
    for coldkey, hotkeys in sorted(coldkeys.items(), key=lambda item: -kept[item[0]]):
        share = kept[coldkey] / total if total else 0.0
        served_share = of(served, hotkeys)
        report.append(
            {
                "coldkey": coldkey,
                "share": share,
                "chars_per_second": kept[coldkey] / window_s,
                "top_hotkey": max(hotkeys, key=lambda h: by_hotkey[h]),
                "served_share": served_share,
                "service_ratio": (
                    served_share / share if share and served_share is not None else None
                ),
                "crawl_share": of(crawled, hotkeys),
                "hotkeys": [
                    {
                        "hotkey": h,
                        "characters": by_hotkey[h],
                        "outcomes": dict(outcomes[h]),
                    }
                    for h in sorted(hotkeys, key=lambda h: -by_hotkey[h])
                ],
            }
        )
    return {
        "tests": len(tests),
        "characters": total,
        "median_gap": statistics.median(gaps) if gaps else None,
        "coldkeys": report,
    }


class Scorer:
    def __init__(self, chain, read_reveal, embedder, db: sqlite3.Connection, dims: int):
        self.chain = chain
        self.read_reveal = read_reveal
        self.embedder = embedder
        self.db = db
        self.dims = dims
        self.flags: dict[int, bool] = {}

    def cursor(self) -> int | None:
        row = self.db.execute("SELECT block FROM cursor").fetchone()
        return row[0] if row else None

    async def scan(self, start: int, end: int) -> None:
        """Finds the tests started by finalized blocks start..end."""
        for low in range(start, end + 1, SCAN_AT_ONCE):
            blocks = range(low, min(low + SCAN_AT_ONCE, end + 1))
            hashes = await asyncio.gather(*(self.chain.block_hash(n) for n in blocks))
            for block, block_hash in zip(blocks, hashes):
                await self.note_block(block, block_hash)
        self.db.execute("INSERT OR REPLACE INTO cursor VALUES (1, ?)", (end,))
        self.db.commit()

    async def note_block(self, block: int, block_hash: bytes) -> None:
        run = range(max(0, block - capacity.QUIET_BLOCKS), block)
        for earlier in run:
            if earlier not in self.flags:
                self.flags[earlier] = capacity.is_candidate(
                    await self.chain.block_hash(earlier)
                )
        self.flags[block] = capacity.is_candidate(block_hash)
        if self.flags[block] and not any(self.flags[n] for n in run):
            log.info("test at block %d", block)
            self.db.execute(
                "INSERT OR IGNORE INTO tests VALUES (?, ?, 'found', ?)",
                (block, block_hash, time.time()),
            )
        for old in [n for n in self.flags if n < run.start]:
            del self.flags[old]

    async def step(self, finalized: int) -> None:
        """Moves every test on as far as the finalized chain allows."""
        pending = self.db.execute(
            "SELECT block, seed, state FROM tests WHERE state IN ('found', 'read') ORDER BY block"
        ).fetchall()
        for block, seed, state in pending:
            test = capacity.Test(block, bytes(seed))
            if await self.chain.block_hash(block) != test.seed:
                self.set_state(block, "left_chain")
                continue
            if state == "found" and finalized >= test.deadline:
                state = await self.read_commitments(test, finalized)
            if state == "read" and finalized >= test.reveal_by + REVEAL_MARGIN_BLOCKS:
                await self.judge(test)

    def set_state(self, block: int, state: str) -> None:
        self.db.execute(
            "UPDATE tests SET state = ?, at = ? WHERE block = ?",
            (state, time.time(), block),
        )
        self.db.commit()

    async def read_commitments(self, test: capacity.Test, finalized: int) -> str:
        """Everyone's commitment as of the deadline block; a later overwrite, made after seeing the samples, is never seen."""
        try:
            registered = await self.chain.registered(test.deadline)
            slots = asyncio.Semaphore(READS_AT_ONCE)

            async def read(hotkey: str) -> Commitment | None:
                async with slots:
                    return await self.chain.commitment(hotkey, test.deadline)

            found = await asyncio.gather(*(read(hotkey) for hotkey in registered))
        except Exception as exc:
            log.warning(
                "test %d: its deadline state cannot be read: %r", test.block, exc
            )
            if finalized - test.deadline < STATE_KEPT_BLOCKS:
                return "found"
            self.set_state(test.block, "missed")
            return "missed"
        for (hotkey, coldkey), commitment in zip(registered.items(), found):
            row, root = commitment_row(test, hotkey, coldkey, commitment)
            self.save(row, root)
        self.set_state(test.block, "read")
        return "read"

    def save(self, row: Row, root: bytes | None = None) -> None:
        self.db.execute(
            """
            INSERT INTO results VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT (block, hotkey) DO UPDATE SET chars = excluded.chars, verdict = excluded.verdict,
                why = excluded.why, worst = excluded.worst, median_gap = excluded.median_gap
            """,
            (
                row.block,
                row.hotkey,
                row.coldkey,
                row.count,
                row.chars,
                row.commit_block,
                root,
                row.verdict,
                row.why,
                row.worst,
                row.median_gap,
            ),
        )

    async def judge(self, test: capacity.Test) -> None:
        sample_hash = await self.chain.block_hash(test.sample_block)
        stored = self.db.execute(
            "SELECT hotkey, coldkey, count, commit_block, root FROM results WHERE block = ? AND verdict != ?",
            (test.block, NO_COMMIT),
        ).fetchall()
        slots = asyncio.Semaphore(READS_AT_ONCE)

        async def one(hotkey, coldkey, count, commit_block, root) -> Row:
            async with slots:
                body = await self.read_reveal(capacity.reveal_key(test.block, hotkey))
                row = Row(
                    test.block, hotkey, coldkey, count, 0, commit_block, NO_REVEAL
                )
                row, opened = judged(
                    test, row, bytes(root), sample_hash, body, self.dims
                )
                if opened is None or self.embedder is None:
                    return row
                picked, vectors = opened
                texts = [capacity.text_at(test.seed, hotkey, i) for i in picked]
                reference = np.array(await self.embedder.embed(texts), dtype=np.float32)
                return with_similarity(
                    test, row, capacity.similarities(vectors, reference)
                )

        rows = await asyncio.gather(*(one(*entry) for entry in stored))
        for row in rows:
            self.save(row)
        self.set_state(test.block, VOID if voided(rows) else "done")
        log.info(
            "test %d: %s",
            test.block,
            {v: sum(r.verdict == v for r in rows) for v in {r.verdict for r in rows}},
        )

    def report(self, since_block: int, served=None, crawled=None) -> dict:
        # A test nobody committed to says nothing of anyone; it must not use up a coldkey's worst test.
        counted = self.db.execute(
            """
            SELECT block FROM tests WHERE state = 'done' AND block >= ?
            AND block IN (SELECT block FROM results WHERE verdict != ?) ORDER BY block
            """,
            (since_block, NO_COMMIT),
        ).fetchall()
        rows = self.db.execute(
            """
            SELECT block, hotkey, coldkey, chars, verdict, median_gap FROM results
            WHERE block IN (SELECT block FROM tests WHERE state = 'done' AND block >= ?) AND verdict != ?
            """,
            (since_block, NO_COMMIT),
        ).fetchall()
        return share_report(rows, [row[0] for row in counted], served, crawled)


async def public_work(
    api: str, http: aiohttp.ClientSession, kind: str, hours: float
) -> dict[str, float] | None:
    """What each hotkey really did, from the task API's public stats; only ever shown beside the test's result."""
    url = f"{api.rstrip('/')}/v1/miners?kind={kind}&hours={int(hours)}"
    try:
        async with http.get(url) as answer:
            answer.raise_for_status()
            miners = (await answer.json())["miners"]
    except (aiohttp.ClientError, asyncio.TimeoutError, KeyError, ValueError) as exc:
        log.warning("reading %s work failed: %r", kind, exc)
        return None
    return {miner["hotkey"]: float(miner["credited"]) for miner in miners}


def public_reveal_reader(bucket: str, http: aiohttp.ClientSession):
    async def read(key: str) -> bytes | None:
        try:
            async with http.get(f"{bucket.rstrip('/')}/{key}") as answer:
                if answer.status != 200:
                    return None
                body = bytearray()
                async for chunk in answer.content.iter_chunked(1 << 16):
                    body += chunk
                    if len(body) > capacity.MAX_REVEAL_BYTES:
                        break
                return bytes(body)
        except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
            log.warning("reading %s failed: %r", key, exc)
            return None

    return read


async def score_once(chain, scorer, http, embedder, window, args) -> None:
    finalized = await chain.finalized()
    cursor = scorer.cursor()
    start = cursor + 1 if cursor is not None else max(0, finalized - args.back_blocks)
    if start <= finalized:
        await scorer.scan(start, finalized)
    await scorer.step(finalized)
    report = scorer.report(
        finalized - window,
        await public_work(args.api, http, "embed", args.hours),
        await public_work(args.api, http, "crawl", args.hours),
    )
    report["finalized"] = finalized
    report["verified"] = embedder is not None
    Path(args.out).write_text(json.dumps(report, indent=1))


async def run(args: argparse.Namespace) -> None:
    model = model_named(args.model)
    db = open_state(args.state)
    embedder = (
        EmbeddingClient(args.model_url, model, args.model_key)
        if args.model_url
        else None
    )
    timeout = aiohttp.ClientTimeout(total=30.0)
    async with (
        CapacityChain(args.network, args.netuid) as chain,
        aiohttp.ClientSession(timeout=timeout) as http,
    ):
        reader = public_reveal_reader(args.bucket, http)
        scorer = Scorer(chain, reader, embedder, db, model.dims)
        window = int(args.hours * 3600 / capacity.SECONDS_PER_BLOCK)
        while True:
            try:
                await score_once(chain, scorer, http, embedder, window, args)
            except Exception:
                if not args.follow:
                    raise
                log.exception("scoring failed; trying again")
            if not args.follow:
                break
            await asyncio.sleep(FOLLOW_S)
    if embedder is not None:
        await embedder.aclose()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--network", default="finney", help="network name or node URL")
    parser.add_argument("--netuid", type=int, default=22)
    parser.add_argument(
        "--bucket", default=PUBLIC_BUCKET, help="the public bucket that holds reveals"
    )
    parser.add_argument(
        "--api",
        default="https://api-22.desearch.ai",
        help="public stats shown beside the result",
    )
    parser.add_argument("--model", default="desearch-4b-v1")
    parser.add_argument(
        "--model-url",
        default="",
        help="your own model server; without it nothing is verified",
    )
    parser.add_argument("--model-key", default="")
    parser.add_argument("--state", default="capacity.db")
    parser.add_argument("--out", default="capacity-scores.json")
    parser.add_argument(
        "--hours",
        type=float,
        default=24.0,
        help="the window scores are worked out over",
    )
    parser.add_argument(
        "--back-blocks",
        type=int,
        default=200,
        help="how far back a first run starts looking for tests",
    )
    parser.add_argument("--follow", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s"
    )
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
