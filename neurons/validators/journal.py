"""Follows the task API's hash-chained log and checks it against our own node: every file's signature and links, every block hash it
decided on, the metagraph it logged, and every head its key ever anchored on chain, read from the blocks themselves. The log is
also replayed with the API's own rules. Anything that does not hold is written down as a fraud proof: the files, entries and chain
reads that show it. Until the log holds together again, `distrust()` says why the API's inputs must not reach the weights.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import sqlite3
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Awaitable, Callable, Optional

import aiohttp

from desearch.journal import (
    BLOCK_SECONDS,
    LATEST_KEY,
    RETENTION_MARGIN_S,
    RETENTION_S,
    Anchor,
    Broken,
    anchor_from_commitment,
    anchors_in_extrinsic,
    blob_key,
    blobs_named,
    blocks_used,
    check_anchor,
    hash_of,
    segment_key,
    signed_by,
    verify_segment,
)
from neurons.validators.custody import as_signers

READ_TIMEOUT = aiohttp.ClientTimeout(total=60.0)
# A validator that starts reads back this many files, about an hour at one file every 30 seconds, or to the newest snapshot, which
# its replay starts from, when that is no further back than `JOIN_MAX_FILES`: snapshots come every six hours, 720 files.
JOIN_BACK = 120
JOIN_MAX_FILES = 800
# Published files are kept here until replayed; the newest few stay for a fraud proof to quote.
KEEP_FILES = 20
REPLAY_TIMEOUT_S = 600.0
# A request the API let in is in a `requests` entry within about this many blocks of arriving.
LOG_LAG_BLOCKS = 5
# The anchors of the last day of blocks are read when a validator starts, then every block as it comes, this many per poll.
RETENTION_BLOCKS = RETENTION_S // BLOCK_SECONDS
SCAN_BLOCKS = 1_200
SCAN_AT_ONCE = 16
# Blocks an anchor needs to be included once sent.
INCLUSION_BLOCKS = 20
# How long a file that is gone, with later ones there, waits for an anchor to show whether the API held its entries.
GAP_WAIT_S = 1_800.0
# A log whose links broke is read again from its newest snapshot this long after, to see whether it holds from there.
REJOIN_AFTER_S = 600.0
# How often a follower without an anchor hotkey to check says so in the validator's log.
UNANCHORED_LOG_S = 600.0
# The log keeps up when it was checked this recently, every file read and replayed, and the chain read to within a few blocks.
STALE_S = 600.0
MAX_BEHIND_FILES = 10
MAX_BEHIND_BLOCKS = 10
# Entries that answer a signed request, by the request's hash: what a witness relay holds the API's answer against.
OUTCOME_KINDS = ("claim", "complete", "abandon", "witness")
OUTCOMES_KEPT_S = 8 * 86400

log = logging.getLogger("validator")

UNANCHORED = (
    "no anchor hotkey is pinned (desearch/trust.py TASK_API_ANCHOR_HOTKEY or --neuron.task_api_anchor), so the log's anchors"
    " are not checked: a log rolled back, forked or taken down would not show"
)


@dataclass
class Problem:
    """One thing about the log that does not hold; `fraud` when the log or the chain proves it."""

    what: str
    severity: str = "fraud"
    seq: Optional[int] = None
    evidence: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "what": self.what,
            "severity": self.severity,
            "seq": self.seq,
            "evidence": self.evidence,
        }


def metagraph_rows(graph) -> list[list]:
    """A metagraph as the API logs it: uid, hotkey, coldkey, permit, total and alpha stake in rao, and the incentive in 65535ths,
    None for every neuron while the chain gives none any."""
    incentives = [round(float(n.incentive) * 65535) for n in graph.neurons]
    known = any(incentives)
    return [
        [
            int(n.uid),
            str(n.hotkey),
            str(n.coldkey),
            bool(n.validator_permit),
            int(n.total_stake.rao),
            int(n.alpha_stake.rao),
            incentive if known else None,
        ]
        for n, incentive in zip(graph.neurons, incentives)
    ]


def logged_rows(listed: list) -> list[list]:
    """The neurons of a `metagraph` entry's blob in the shape of `metagraph_rows`."""
    return [
        [
            int(row[0]),
            str(row[1]),
            str(row[2] or ""),
            bool(row[3]),
            int(row[4]),
            int(row[5]),
            None if row[6] is None else round(float(row[6]) * 65535),
        ]
        for row in listed
    ]


class JournalFollower:
    def __init__(
        self,
        http: aiohttp.ClientSession,
        storage_url: str,
        signer: str,
        folder: Path,
        block_hash: Callable[[int], Awaitable[Optional[str]]],
        anchor: Callable[[], Awaitable[Optional[Anchor]]],
        replay_bin: str = "",
        *,
        chain_head: Optional[Callable[[], Awaitable[int]]] = None,
        block_extrinsics: Optional[
            Callable[[int], Awaitable[Optional[list[bytes]]]]
        ] = None,
        anchor_hotkey: str = "",
        metagraph_at: Optional[Callable[[int, int], Awaitable[list[list]]]] = None,
        clock: Callable[[], float] = time.time,
    ):
        self.http = http
        self.storage_url = storage_url.rstrip("/")
        # One address, or the custody chain naming who signed which entries.
        self.signers = as_signers(signer)
        self.folder = Path(folder)
        self.block_hash = block_hash
        self.anchor = anchor
        self.replay_bin = replay_bin
        # The head of our own node, a block's extrinsics and the metagraph at a block, all read from it.
        self.chain_head = chain_head
        self.block_extrinsics = block_extrinsics
        self.anchor_key = public_key_of(anchor_hotkey) if anchor_hotkey else None
        self.metagraph_at = metagraph_at
        self.clock = clock
        for sub in ("seq", "blobs", "fraud"):
            (self.folder / sub).mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.folder / "journal.db")
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS state (name TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS hashes (seq INTEGER PRIMARY KEY, hash TEXT NOT NULL, segment INTEGER NOT NULL);
            CREATE TABLE IF NOT EXISTS blocks (block INTEGER PRIMARY KEY, hash TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS requests (hash TEXT PRIMARY KEY, seq INTEGER NOT NULL, block INTEGER);
            CREATE TABLE IF NOT EXISTS request_entries (seq INTEGER PRIMARY KEY, block INTEGER);
            CREATE TABLE IF NOT EXISTS rounds (round_id TEXT PRIMARY KEY, view TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS chain_anchors (
                block   INTEGER NOT NULL,
                seq     INTEGER NOT NULL,
                head    TEXT NOT NULL,
                matched INTEGER,
                PRIMARY KEY (block, seq, head)
            );
            CREATE TABLE IF NOT EXISTS reported (what TEXT PRIMARY KEY, at REAL NOT NULL);
            CREATE TABLE IF NOT EXISTS outcomes (
                request TEXT NOT NULL, seq INTEGER NOT NULL, at REAL NOT NULL, entry TEXT NOT NULL,
                PRIMARY KEY (request, seq)
            );
            """)

    def state(self, name: str):
        row = self.db.execute(
            "SELECT value FROM state WHERE name = ?", (name,)
        ).fetchone()
        return json.loads(row[0]) if row else None

    def set_state(self, name: str, value) -> None:
        self.db.execute(
            "INSERT OR REPLACE INTO state VALUES (?, ?)", (name, json.dumps(value))
        )
        self.db.commit()

    def hash_at(self, seq: int) -> Optional[str]:
        row = self.db.execute(
            "SELECT hash FROM hashes WHERE seq = ?", (seq,)
        ).fetchone()
        return row[0] if row else None

    def included(self, request: str) -> Optional[tuple[int, Optional[int]]]:
        """The entry and block a signed request reached the log at, by its hash; None while it has not."""
        row = self.db.execute(
            "SELECT seq, block FROM requests WHERE hash = ?", (request,)
        ).fetchone()
        return (row[0], row[1]) if row else None

    def round_view(self, round_id: str) -> Optional[dict]:
        """A closed crawl round's view as the API settled every upload of it: its loadability, coverage gate and neutral
        hosts, from the round's `close` entry in the signed log; None while it is open.
        """
        row = self.db.execute(
            "SELECT view FROM rounds WHERE round_id = ?", (round_id,)
        ).fetchone()
        return json.loads(row[0]) if row else None

    def round_views(self) -> dict[str, dict]:
        """Every closed round's settled view this follower has read, by round id."""
        return {
            round_id: json.loads(view)
            for round_id, view in self.db.execute("SELECT round_id, view FROM rounds")
        }

    def outcomes(self, request: str) -> list[dict]:
        """The entries answering a signed request, by its hash, that an anchor on chain covers."""
        rows = self.db.execute(
            "SELECT entry FROM outcomes WHERE request = ? AND seq <= ? ORDER BY seq",
            (request, self.anchored_through()),
        ).fetchall()
        return [json.loads(entry) for (entry,) in rows]

    def anchored_through(self) -> int:
        """The last entry an anchor on chain covers that this log agrees with."""
        row = self.db.execute(
            "SELECT COALESCE(MAX(seq), 0) FROM chain_anchors WHERE matched = 1"
        ).fetchone()
        return max(int(self.state("anchored") or 0), int(row[0]))

    def covered_block(self) -> int:
        """The newest block whose requests the anchored log has listed: a request let in before it is in the log."""
        row = self.db.execute(
            "SELECT COALESCE(MAX(block), 0) FROM request_entries WHERE seq <= ?",
            (self.anchored_through(),),
        ).fetchone()
        return int(row[0] or 0)

    def distrust(self) -> Optional[str]:
        """Why the API's budgets, verdicts, shares and upload log must not reach the weights now; None while its log holds together
        and keeps up: every file read, replayed with its rules from a snapshot and checked against our own node.
        """
        fraud = self.state("fraud")
        if fraud:
            return f"the task API's log shows fraud at entry {fraud['seq']}: {fraud['what']}"
        if self.state("broken"):
            return f"the task API's log does not hold together: {self.state('broken')}"
        if self.anchor_key is None:
            return UNANCHORED
        if not self.replay_bin:
            return "the task API's log is not replayed (--neuron.no_replay)"
        checked = self.state("checked_at")
        if checked is None or self.clock() - checked > STALE_S:
            return "the task API's log has not been read and replayed lately"
        behind = int(self.state("files_behind") or 0)
        if behind > MAX_BEHIND_FILES:
            return f"the log is read {behind} files behind what the API published"
        replayed = self.state("replay") or {}
        if not replayed.get("started"):
            return "the replay waits for a snapshot of the accounts to start from"
        if int(replayed.get("last") or 0) < int(self.state("last") or 0):
            return "the replay is behind the log read"
        if (
            self.anchor_key is not None
            and int(self.state("scan_behind") or 0) > MAX_BEHIND_BLOCKS
        ):
            return "the anchors on chain are not read up to the head yet"
        return None

    def status(self) -> dict:
        """What this follower vouches for, published beside its proofs: whether the log's anchors are checked at all, and why the
        API's inputs are kept out of the weights, if they are."""
        return {
            "anchors_checked": self.anchor_key is not None,
            "replayed": bool(self.replay_bin),
            "distrust": self.distrust(),
            "fraud": self.state("fraud"),
            "last": self.state("last"),
            "anchored_through": self.anchored_through(),
        }

    def fraud_spans(self) -> list[tuple[float, float]]:
        """The times the log showed fraud, from the entry it was found at to when the log held together again: uploads logged then
        are never paid from the log."""
        return [
            (start, end if end is not None else self.clock())
            for start, end in self.state("fraud_spans") or []
        ]

    async def get(self, key: str):
        async with self.http.get(
            f"{self.storage_url}/{key}", timeout=READ_TIMEOUT
        ) as response:
            # A store that keeps its bucket private answers 403, not 404, for an object it does not have.
            if response.status in (403, 404):
                return None
            response.raise_for_status()
            return await response.read()

    async def get_json(self, key: str):
        found = await self.get(key)
        return None if found is None else json.loads(found)

    async def poll(self) -> list[Problem]:
        """Reads the files published since the last poll and checks them; returns what does not hold."""
        problems: list[Problem] = []
        latest = await self.get_json(LATEST_KEY)
        if latest is None:
            return problems
        allowed = self.signers.for_journal(int(latest.get("last") or 0))
        if not any(signed_by(latest, signer) for signer in allowed):
            return self.reported(
                [
                    Problem(
                        f"{LATEST_KEY} is not signed by {' or '.join(allowed) or 'any holder'}",
                        evidence={"latest": latest},
                    )
                ]
            )
        if self.state("broken"):
            # Read again from the newest snapshot after a while: the log is trusted again only once it holds from there.
            if self.clock() - float(self.state("broken_at") or 0) < REJOIN_AFTER_S:
                return problems
            self.db.execute("DELETE FROM state WHERE name IN ('broken', 'broken_at')")
            await self.rejoin(latest, after=None)
        if self.anchor_key is None:
            # Not a passing check: none is made. Said every few minutes, and published as the follower's status.
            said = float(self.state("unanchored_said_at") or 0)
            if self.clock() - said >= UNANCHORED_LOG_S:
                log.error("task API log: %s", UNANCHORED)
                self.set_state("unanchored_said_at", self.clock())
        problems += await self.scan_chain()
        number = self.state("segment")
        joining = number is None or bool(self.state("joining"))
        if number is None:
            number = join_from(latest)
        while number <= int(latest["segment"]):
            segment = await self.get_json(segment_key(number))
            if segment is None:
                if number < int(latest["segment"]):
                    found, then = await self.missing(number, latest, joining)
                    problems += found
                    if then == "rejoin":
                        await self.rejoin(latest, after=number)
                    elif then == "broken":
                        self.break_off(found[0].what)
                break
            self.db.execute("DELETE FROM state WHERE name = 'gap_since'")
            found = await self.take(segment, joining)
            problems += found
            if any(p.severity == "fraud" for p in found):
                break
            joining = False
            number += 1
            self.set_state("segment", number)
            self.set_state("joining", False)
        self.set_state(
            "files_behind",
            max(0, int(latest["segment"]) - int(self.state("segment") or 0) + 1),
        )
        problems += self.check_history()
        problems += await self.check_anchor()
        replayed = True
        if self.replay_bin and not self.state("broken"):
            found, replayed = await self.replay()
            problems += found
        problems = self.reported(problems)
        for problem in problems:
            self.write_proof(problem)
        self.note(problems)
        if replayed and not self.state("broken"):
            self.set_state("checked_at", self.clock())
        self.prune()
        return problems

    def reported(self, problems: list[Problem]) -> list[Problem]:
        """Leaves out a fraud already reported, so a proof is written once however many polls see it."""
        fresh = []
        for problem in problems:
            if problem.severity == "fraud":
                key = f"{problem.seq}:{problem.what}"
                if self.db.execute(
                    "SELECT 1 FROM reported WHERE what = ?", (key,)
                ).fetchone():
                    continue
                self.db.execute(
                    "INSERT INTO reported VALUES (?, ?)", (key, self.clock())
                )
            fresh.append(problem)
        self.db.commit()
        return fresh

    def note(self, problems: list[Problem]) -> None:
        """A fraud makes the log untrusted from the entry it shows until the replay, started again from a snapshot past it, holds
        through an anchor on chain; then the log is trusted again."""
        frauds = [p for p in problems if p.severity == "fraud"]
        fraud = self.state("fraud")
        if frauds:
            seq = max(p.seq or 0 for p in frauds)
            if fraud is None:
                # From the earliest entry shown to be fraud, when the files kept still hold it.
                first = min(p.seq or seq for p in frauds)
                spans = self.state("fraud_spans") or []
                spans.append([self.entry_at(first) or self.clock(), None])
                self.set_state("fraud_spans", spans)
            self.set_state(
                "fraud",
                {
                    "seq": max(seq, int((fraud or {}).get("seq") or 0)),
                    "what": frauds[0].what,
                    "at": self.clock(),
                },
            )
            # The replay starts again at the first snapshot past the fraud, from files after it.
            (self.folder / "replay.db").unlink(missing_ok=True)
            self.db.execute("DELETE FROM state WHERE name = 'replay'")
            for path in (self.folder / "seq").glob("*.json"):
                if int(json.loads(path.read_text()).get("last") or 0) <= seq:
                    path.unlink()
            self.db.commit()
            return
        if fraud is None:
            return
        replayed = self.state("replay") or {}
        since = replayed.get("since")
        if (
            since is not None
            and since > int(fraud["seq"])
            and self.anchored_through() >= since
        ):
            spans = self.state("fraud_spans") or []
            if spans and spans[-1][1] is None:
                spans[-1][1] = self.clock()
            self.set_state("fraud_spans", spans)
            self.db.execute("DELETE FROM state WHERE name = 'fraud'")
            self.db.commit()
            log.warning(
                "The task API's log holds together again from entry %d: its inputs reach the weights again",
                since,
            )

    def entry_at(self, seq: int) -> Optional[float]:
        """When entry `seq` says it was made, from the files kept."""
        for path in (self.folder / "seq").glob("*.json"):
            segment = json.loads(path.read_text())
            if int(segment.get("first") or 0) <= seq <= int(segment.get("last") or 0):
                for entry in segment.get("entries") or []:
                    if entry.get("seq") == seq:
                        return float(entry.get("at") or 0) or None
        return None

    def break_off(self, what: str) -> None:
        self.set_state("broken", what)
        self.set_state("broken_at", self.clock())

    async def missing(
        self, number: int, latest: dict, joining: bool
    ) -> tuple[list[Problem], str]:
        """A file gone while later ones are published. Within the bucket's retention and covered by an anchor, the API held its
        entries and took them down: fraud. Without the anchors to tell, the follower reads on from a recent file.
        """
        if (
            joining
            or self.state("last") is None
            or self.anchor_key is None
            or self.chain_head is None
        ):
            return [
                Problem(
                    f"log file {number} is gone while {latest['segment']} is published; reading on from a recent file",
                    severity="suspect",
                    evidence={"latest": latest},
                )
            ], "rejoin"
        first = int(self.state("last")) + 1
        covering = self.db.execute(
            "SELECT seq, block, head FROM chain_anchors WHERE seq >= ? ORDER BY block LIMIT 1",
            (first,),
        ).fetchone()
        if covering is None:
            since = self.state("gap_since")
            if since is None:
                self.set_state("gap_since", self.clock())
            elif self.clock() - since >= GAP_WAIT_S:
                return [
                    Problem(
                        f"log file {number} is gone and no anchor covers its entries; reading on from a recent file",
                        severity="suspect",
                        evidence={"latest": latest, "first": first},
                    )
                ], "rejoin"
            return [
                Problem(
                    f"log file {number} is gone while {latest['segment']} is published; waiting for an anchor to cover entry {first}",
                    severity="suspect",
                    evidence={"latest": latest, "first": first},
                )
            ], "wait"
        seq, block, head = covering
        before = self.db.execute(
            "SELECT MAX(block) FROM chain_anchors WHERE seq < ?", (first,)
        ).fetchone()[0]
        now_block = await self.chain_head()
        # Entries past an anchored head were made after that anchor was sent, at most `INCLUSION_BLOCKS` before it landed.
        age_s = (
            None
            if before is None
            else (now_block - before + INCLUSION_BLOCKS) * BLOCK_SECONDS
        )
        evidence = {
            "latest": latest,
            "file": number,
            "first": first,
            "anchor": {"seq": seq, "head": head, "block": block},
            "written_after_block": before,
            "head_block": now_block,
            "retention_s": RETENTION_S,
        }
        if age_s is not None and age_s < RETENTION_S - RETENTION_MARGIN_S:
            return [
                Problem(
                    f"log file {number} (entries from {first}) is gone although the chain anchors entry {seq} at block {block}"
                    f" and the file was written after block {before}, inside the bucket's retention: the API took it down",
                    seq=first,
                    evidence=evidence,
                )
            ], "broken"
        return [
            Problem(
                f"log file {number} is gone, older than the bucket's retention; reading on from a recent file",
                severity="suspect",
                evidence=evidence,
            )
        ], "rejoin"

    async def take(self, segment: dict, joining: bool) -> list[Problem]:
        last = int(self.state("last") or 0)
        head = self.state("head")
        if joining or head is None:
            last, head = int(segment["first"]) - 1, segment["prev"]
        try:
            last, head = self.verified(segment, last, head)
        except Broken as broken:
            self.break_off(str(broken))
            return [
                Problem(
                    str(broken),
                    seq=last + 1,
                    evidence={"segment": segment, "after": {"seq": last, "head": head}},
                )
            ]
        entries = segment["entries"]
        problems = await self.check_blocks(entries)
        named = blobs_named(entries) if self.replay_bin else set()
        for entry in entries:
            given = entry.get("input") or {}
            if entry.get("kind") == "requests":
                listed = given.get("requests")
                if isinstance(listed, dict) and listed.get("blob"):
                    named.add(listed["blob"])
            if entry.get("kind") == "metagraph":
                listed = given.get("neurons")
                if isinstance(listed, dict) and listed.get("blob"):
                    named.add(listed["blob"])
        for sha in named:
            path = self.folder / "blobs" / sha
            if not path.exists():
                found = await self.get(blob_key(sha))
                if found is not None:
                    path.write_bytes(found)
        for entry in entries:
            if entry.get("kind") == "requests":
                self.index_requests(entry)
            elif entry.get("kind") == "close":
                self.index_round(entry)
            elif entry.get("kind") == "metagraph":
                problems += await self.check_metagraph(entry)
            elif entry.get("kind") in OUTCOME_KINDS:
                self.index_outcome(entry)
        self.db.executemany(
            "INSERT OR REPLACE INTO hashes VALUES (?, ?, ?)",
            [(entry["seq"], hash_of(entry), segment["segment"]) for entry in entries],
        )
        self.db.commit()
        (self.folder / "seq" / Path(segment_key(segment["segment"])).name).write_text(
            json.dumps(segment)
        )
        self.set_state("last", last)
        self.set_state("head", head)
        return problems

    def verified(self, segment: dict, last: int, head: str) -> tuple[int, str]:
        """A file checked against whoever held the log for its entries; a holder's signature with broken links says more than none."""
        allowed = self.signers.for_journal(int(segment.get("last") or 0))
        problems = []
        for signer in allowed:
            try:
                return verify_segment(segment, signer, last, head)
            except Broken as broken:
                problems.append(broken)
        named = [p for p in problems if "not signed" not in str(p)]
        found = named or problems
        if found:
            raise found[0]
        raise Broken(f"file {segment.get('segment')} has no holder to be signed by")

    def index_round(self, entry: dict) -> None:
        round_id = (entry.get("input") or {}).get("round_id")
        view = (entry.get("output") or {}).get("round")
        if round_id and isinstance(view, dict):
            self.db.execute(
                "INSERT OR REPLACE INTO rounds VALUES (?, ?)",
                (round_id, json.dumps(view)),
            )

    def index_requests(self, entry: dict) -> None:
        given = entry.get("input") or {}
        listed = given.get("requests")
        if isinstance(listed, dict):
            path = self.folder / "blobs" / str(listed.get("blob"))
            listed = json.loads(path.read_bytes()) if path.exists() else []
        self.db.executemany(
            "INSERT OR IGNORE INTO requests VALUES (?, ?, ?)",
            [(row[0], entry["seq"], given.get("block")) for row in listed or [] if row],
        )
        self.db.execute(
            "INSERT OR REPLACE INTO request_entries VALUES (?, ?)",
            (entry["seq"], given.get("block")),
        )

    def index_outcome(self, entry: dict) -> None:
        request = (entry.get("input") or {}).get("request")
        named = request.get("request") if isinstance(request, dict) else None
        if isinstance(named, str):
            self.db.execute(
                "INSERT OR REPLACE INTO outcomes VALUES (?, ?, ?, ?)",
                (named, entry["seq"], float(entry.get("at") or 0), json.dumps(entry)),
            )

    async def check_blocks(self, entries: list[dict]) -> list[Problem]:
        """Every block hash the API decided on, against our own node."""
        problems = []
        for used in blocks_used(entries):
            row = self.db.execute(
                "SELECT hash FROM blocks WHERE block = ?", (used.block,)
            ).fetchone()
            ours = row[0] if row else await self.block_hash(used.block)
            if ours is None:
                continue
            if row is None:
                self.db.execute(
                    "INSERT OR IGNORE INTO blocks VALUES (?, ?)", (used.block, ours)
                )
            if ours.lower() != used.hash.lower():
                problems.append(
                    Problem(
                        f"entry {used.seq} decided on block {used.block} with hash {used.hash}; our node has {ours}",
                        seq=used.seq,
                        evidence={
                            "block": used.block,
                            "logged": used.hash,
                            "node": ours,
                        },
                    )
                )
        self.db.commit()
        return problems

    async def check_metagraph(self, entry: dict) -> list[Problem]:
        """Who could vote and what capped each budget, as the API logged it, against the metagraph our own node has at that block."""
        if self.metagraph_at is None:
            return []
        given = entry.get("input") or {}
        sha = (given.get("neurons") or {}).get("blob")
        path = self.folder / "blobs" / str(sha)
        if given.get("block") is None or not path.exists():
            return [
                Problem(
                    f"entry {entry['seq']} logs a metagraph without the block it was read at or its neurons",
                    seq=entry["seq"],
                    evidence={"entry": entry},
                )
            ]
        block, netuid = int(given["block"]), int(given.get("netuid") or 0)
        try:
            ours = await self.metagraph_at(netuid, block)
        except Exception as error:
            # A pruned node without an archive to fall back on keeps no state this old.
            return [
                Problem(
                    f"the metagraph of entry {entry['seq']} was not checked: our node has no state at block {block} ({error!r})",
                    severity="suspect",
                    seq=entry["seq"],
                )
            ]
        logged = logged_rows(json.loads(path.read_bytes()))
        if logged == ours:
            return []
        theirs = {row[0]: row for row in logged}
        node = {row[0]: row for row in ours}
        differ = sorted(
            uid for uid in set(theirs) | set(node) if theirs.get(uid) != node.get(uid)
        )
        return [
            Problem(
                f"entry {entry['seq']} logs a metagraph at block {block} that our node does not have: {len(differ)} neurons differ",
                seq=entry["seq"],
                evidence={
                    "block": block,
                    "netuid": netuid,
                    "logged": [theirs.get(uid) for uid in differ[:20]],
                    "node": [node.get(uid) for uid in differ[:20]],
                },
            )
        ]

    async def scan_chain(self) -> list[Problem]:
        """Reads every anchor the API's hotkey sent from the blocks themselves, the last day's when starting, then each new block's."""
        if (
            self.anchor_key is None
            or self.block_extrinsics is None
            or self.chain_head is None
        ):
            return []
        head = await self.chain_head()
        scanned = self.state("scanned")
        start = (
            int(scanned) + 1 if scanned is not None else max(1, head - RETENTION_BLOCKS)
        )
        stop = min(head, start + SCAN_BLOCKS - 1)
        reached = None
        for low in range(start, stop + 1, SCAN_AT_ONCE):
            blocks = list(range(low, min(stop, low + SCAN_AT_ONCE - 1) + 1))
            bodies = await asyncio.gather(*(self.block_extrinsics(b) for b in blocks))
            for block, extrinsics in zip(blocks, bodies):
                if extrinsics is None:
                    break
                self.db.executemany(
                    "INSERT OR IGNORE INTO chain_anchors (block, seq, head) VALUES (?, ?, ?)",
                    [
                        (block, seq, anchored)
                        for extrinsic in extrinsics
                        for seq, anchored in anchors_in_extrinsic(
                            extrinsic, self.anchor_key
                        )
                    ],
                )
                reached = block
            if reached != blocks[-1]:
                break
        if reached is not None:
            self.set_state("scanned", reached)
        self.set_state("scan_behind", head - int(self.state("scanned") or 0))
        self.db.execute(
            "DELETE FROM chain_anchors WHERE block < ?", (head - 2 * RETENTION_BLOCKS,)
        )
        self.db.commit()
        return []

    def check_history(self) -> list[Problem]:
        """Every head the API's key anchored, in block order: never an entry before one anchored earlier, never two heads for one
        entry, and each the log's own hash at that entry. A copy that rolled back or forked its log shows here, to a validator
        that started after it did."""
        problems = []
        highest: Optional[tuple[int, int, str]] = None
        heads: dict[int, tuple[str, int]] = {}
        rows = self.db.execute(
            "SELECT block, seq, head, matched FROM chain_anchors ORDER BY block, seq"
        ).fetchall()
        for block, seq, head, matched in rows:
            if highest is not None and seq < highest[0]:
                problems.append(
                    Problem(
                        f"the anchor went back from entry {highest[0]} (block {highest[1]}) to {seq} (block {block}):"
                        " the API rolled its log back",
                        seq=highest[0],
                        evidence={
                            "before": {
                                "seq": highest[0],
                                "block": highest[1],
                                "head": highest[2],
                            },
                            "after": {"seq": seq, "block": block, "head": head},
                        },
                    )
                )
            if seq in heads and heads[seq][0] != head:
                problems.append(
                    Problem(
                        f"the chain anchors two heads for entry {seq} (blocks {heads[seq][1]} and {block}): the log forked",
                        seq=seq,
                        evidence={
                            "first": {
                                "head": heads[seq][0],
                                "block": heads[seq][1],
                            },
                            "second": {"head": head, "block": block},
                        },
                    )
                )
            heads.setdefault(seq, (head, block))
            if highest is None or seq > highest[0]:
                highest = (seq, block, head)
            if matched is None:
                ours = self.hash_at(seq)
                if ours is None:
                    continue
                self.db.execute(
                    "UPDATE chain_anchors SET matched = ? WHERE block = ? AND seq = ? AND head = ?",
                    (int(ours == head), block, seq, head),
                )
                if ours != head:
                    problems.append(
                        Problem(
                            f"the chain anchors {head} at entry {seq} (block {block}), the log read has {ours}:"
                            " the log was rewritten or forked",
                            seq=seq,
                            evidence={
                                "anchor": {"seq": seq, "head": head, "block": block},
                                "log": ours,
                            },
                        )
                    )
        self.db.commit()
        return problems

    def verified_anchors(self) -> list[dict]:
        """The anchors on chain whose head is the log's own, for the replay to count a round's commitment as anchored by."""
        return [
            {"seq": seq, "head": head, "block": block}
            for block, seq, head in self.db.execute(
                "SELECT block, seq, head FROM chain_anchors WHERE matched = 1 ORDER BY block"
            )
        ]

    async def check_anchor(self) -> list[Problem]:
        anchor = await self.anchor()
        if anchor is None:
            return []
        seen = self.state("anchor")
        if seen and (
            anchor.seq < seen["seq"]
            or (anchor.seq == seen["seq"] and anchor.head != seen["head"])
        ):
            return [
                Problem(
                    f"the anchor went back from entry {seen['seq']} to {anchor.seq}",
                    seq=seen["seq"],
                    evidence={"before": seen, "now": anchor.__dict__},
                )
            ]
        self.set_state("anchor", anchor.__dict__)
        wrong = check_anchor(anchor, self.hash_at)
        if wrong:
            self.break_off(wrong)
            return [
                Problem(
                    wrong,
                    seq=anchor.seq,
                    evidence={
                        "anchor": anchor.__dict__,
                        "log": self.hash_at(anchor.seq),
                    },
                )
            ]
        if self.hash_at(anchor.seq) is not None:
            self.set_state(
                "anchored", max(anchor.seq, int(self.state("anchored") or 0))
            )
        return []

    async def replay(self) -> tuple[list[Problem], bool]:
        """Runs the API's replay over the files read, with the anchors our node shows; its state lives beside them, so each run goes
        on from the last. Returns what it found and whether it ran to the end."""
        anchors = self.folder / "anchors.json"
        anchors.write_text(json.dumps(self.verified_anchors()))
        command = [
            self.replay_bin,
            "--dir",
            str(self.folder),
            "--state",
            str(self.folder / "replay.db"),
            *[part for signer in self.signers.every() for part in ("--signer", signer)],
            "--anchors",
            str(anchors),
            "--join",
        ]
        try:
            process = await asyncio.create_subprocess_exec(
                *command, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
            )
            out, err = await asyncio.wait_for(process.communicate(), REPLAY_TIMEOUT_S)
        except (OSError, asyncio.TimeoutError) as error:
            log.warning("the log replay did not run: %r", error)
            return [], False
        problems, summary = [], None
        for line in out.decode().splitlines():
            found = json.loads(line)
            if "severity" in found:
                problems.append(
                    Problem(found["what"], found["severity"], found.get("seq"), found)
                )
            elif "last" in found:
                summary = found
        if summary is not None:
            self.set_state("replay", summary)
        if process.returncode not in (0, None) and not problems:
            log.warning("the log replay failed: %s", err.decode()[-2000:])
            return problems, False
        return problems, summary is not None or bool(problems)

    async def rejoin(self, latest: dict, after: Optional[int]) -> None:
        """Starts again from a later file: from the newest snapshot when it lies past file `after`, else from the first file there
        after it (the newest after a break). The chain is trusted from there until an anchor covers it, and the replay starts
        again at the snapshot."""
        start = int(latest["segment"])
        snapshot = latest.get("snapshot")
        if snapshot is not None and (after or 0) < int(snapshot) <= start:
            start = int(snapshot)
        elif after is not None:
            for number in range(after + 1, start + 1):
                if await self.get(segment_key(number)) is not None:
                    start = number
                    break
        for name in ("last", "head", "replay"):
            self.db.execute("DELETE FROM state WHERE name = ?", (name,))
        self.set_state("segment", start)
        self.set_state("joining", True)
        (self.folder / "replay.db").unlink(missing_ok=True)
        for path in (self.folder / "seq").glob("*.json"):
            path.unlink()

    def prune(self) -> None:
        """Files the replay has gone past go, but for the newest few; without the replay, only those."""
        files = sorted((self.folder / "seq").glob("*.json"))
        for path in files[:-KEEP_FILES]:
            path.unlink(missing_ok=True)
        self.db.execute(
            "DELETE FROM outcomes WHERE at < ?", (time.time() - OUTCOMES_KEPT_S,)
        )
        self.db.commit()

    def write_proof(self, problem: Problem) -> None:
        if problem.severity != "fraud":
            return
        name = f"{problem.seq or 0:012d}-{int(self.clock())}.json"
        proof = {
            "problem": problem.as_dict(),
            "signer": str(self.signers),
            "anchor": self.state("anchor"),
            "storage_url": self.storage_url,
            "files": sorted(p.name for p in (self.folder / "seq").glob("*.json"))[
                -KEEP_FILES:
            ],
            # The signed file holding the entry, so the proof stands after the bucket lets it go.
            "segment": self.segment_of(problem.seq),
        }
        (self.folder / "fraud" / name).write_text(json.dumps(proof, indent=2))
        log.error(
            "task API log: %s (proof in %s)", problem.what, self.folder / "fraud" / name
        )

    def segment_of(self, seq: Optional[int]) -> Optional[dict]:
        if seq is None:
            return None
        for path in (self.folder / "seq").glob("*.json"):
            segment = json.loads(path.read_text())
            if int(segment.get("first") or 0) <= seq <= int(segment.get("last") or 0):
                return segment
        return None

    def proofs(self, since: float = 0.0) -> list[dict]:
        """The fraud proofs written since `since`, oldest first, as a relay publishes them."""
        found = []
        for path in sorted((self.folder / "fraud").glob("*.json")):
            if path.stat().st_mtime >= since:
                found.append(json.loads(path.read_text()))
        return found


def replay_problem(path: str, cargo_toml: Optional[Path] = None) -> Optional[str]:
    """Why the replay tool at `path` cannot replay the log with this release's rules, or None: it runs, and it is the version the
    release's own `task-api/Cargo.toml` builds."""
    if not path:
        return "no replay tool is set (--neuron.replay_bin)"
    try:
        ran = subprocess.run(
            [path, "--version"], capture_output=True, text=True, timeout=30
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        return f"the replay tool {path} does not run: {error}"
    words = ran.stdout.split()
    if ran.returncode != 0 or len(words) != 2 or words[0] != "task-api-replay":
        return f"{path} is not the task API's replay tool: {(ran.stdout + ran.stderr).strip()[:200]!r}"
    wanted = crate_version(cargo_toml)
    if wanted is not None and words[1] != wanted:
        return f"the replay tool {path} is version {words[1]}, this release's rules are {wanted}: build it again"
    return None


def crate_version(cargo_toml: Optional[Path] = None) -> Optional[str]:
    """The task API's version in this checkout, None outside one."""
    path = cargo_toml or Path(__file__).resolve().parents[2] / "task-api" / "Cargo.toml"
    try:
        text = path.read_text()
    except OSError:
        return None
    package = text.split("[package]", 1)[-1].split("\n[", 1)[0]
    found = re.search(r'^version\s*=\s*"([^"]+)"', package, re.MULTILINE)
    return found.group(1) if found else None


def join_from(latest: dict) -> int:
    """The file a validator that starts reads from: the newest snapshot's, so its replay can start at once, when that is near."""
    newest = int(latest["segment"])
    snapshot = latest.get("snapshot")
    if snapshot is not None and newest - JOIN_MAX_FILES <= int(snapshot) <= newest:
        return int(snapshot)
    return max(1, newest - JOIN_BACK)


def public_key_of(address: str) -> bytes:
    from bittensor.wallets import Keypair

    return bytes(Keypair(ss58_address=address).public_key)


class JournalLogView:
    """What a censorship check asks of the log: a relayed request is in it when a `requests` entry names it, and is
    not when the anchored log already lists requests from blocks after it; otherwise the API's signed receipts
    answer (`fallback`, the witness relays' `ApiLog`)."""

    def __init__(
        self,
        follower: Callable[[], Optional["JournalFollower"]],
        head,
        after_blocks: int,
        fallback=None,
        lag_blocks: int = LOG_LAG_BLOCKS,
    ):
        self.follower = follower
        self.head = head
        self.after_blocks = after_blocks
        self.fallback = fallback
        self.lag_blocks = lag_blocks

    async def included(self, request: str) -> bool | None:
        follower = self.follower()
        if follower is not None:
            if follower.included(request) is not None:
                return True
            head = self.head.block
            # A check asks only of requests relayed `after_blocks` or more before the head.
            if (
                head is not None
                and follower.covered_block()
                >= head - self.after_blocks + self.lag_blocks
            ):
                return False
        return await self.fallback.included(request) if self.fallback else None
