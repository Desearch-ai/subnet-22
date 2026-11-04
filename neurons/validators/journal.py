"""Follows the task API's hash-chained log and checks it against our own node: every file's signature and links, every block hash it
decided on, and every head it anchored on chain. With the replay tool installed, the log is also replayed with the API's own rules.
Anything that does not hold is written down as a fraud proof: the files, entries and chain reads that show it.
"""

from __future__ import annotations

import asyncio
import json
import logging
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Awaitable, Callable, Optional

import aiohttp

from desearch.journal import (
    LATEST_KEY,
    Anchor,
    Broken,
    anchor_from_commitment,
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
# A validator that starts reads back this many files, about an hour at one file every 30 seconds.
JOIN_BACK = 120
# Published files are kept here until replayed; the newest few stay for a fraud proof to quote.
KEEP_FILES = 20
REPLAY_TIMEOUT_S = 600.0
# A request the API let in is in a `requests` entry within about this many blocks of arriving.
LOG_LAG_BLOCKS = 5

log = logging.getLogger("validator")


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
    ):
        self.http = http
        self.storage_url = storage_url.rstrip("/")
        # One address, or the custody chain naming who signed which entries.
        self.signers = as_signers(signer)
        self.folder = Path(folder)
        self.block_hash = block_hash
        self.anchor = anchor
        self.replay_bin = replay_bin
        for sub in ("seq", "blobs", "fraud"):
            (self.folder / sub).mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.folder / "journal.db")
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS state (name TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS hashes (seq INTEGER PRIMARY KEY, hash TEXT NOT NULL, segment INTEGER NOT NULL);
            CREATE TABLE IF NOT EXISTS blocks (block INTEGER PRIMARY KEY, hash TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS requests (hash TEXT PRIMARY KEY, seq INTEGER NOT NULL, block INTEGER);
            CREATE TABLE IF NOT EXISTS request_entries (seq INTEGER PRIMARY KEY, block INTEGER);
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

    def anchored_through(self) -> int:
        """The last entry an anchor on chain covers that this log agrees with."""
        return int(self.state("anchored") or 0)

    def covered_block(self) -> int:
        """The newest block whose requests the anchored log has listed: a request let in before it is in the log."""
        row = self.db.execute(
            "SELECT COALESCE(MAX(block), 0) FROM request_entries WHERE seq <= ?",
            (self.anchored_through(),),
        ).fetchone()
        return int(row[0] or 0)

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
            return [
                Problem(
                    f"{LATEST_KEY} is not signed by {' or '.join(allowed) or 'any holder'}",
                    evidence={"latest": latest},
                )
            ]
        if self.state("broken"):
            return problems
        number = self.state("segment")
        joining = number is None or bool(self.state("joining"))
        if number is None:
            number = max(1, int(latest["segment"]) - JOIN_BACK)
        while number <= int(latest["segment"]):
            segment = await self.get_json(segment_key(number))
            if segment is None:
                if number < int(latest["segment"]):
                    # Files go after a day; one gone while later ones are there leaves a gap only the anchors can bridge.
                    problems.append(
                        Problem(
                            f"log file {number} is gone while {latest['segment']} is published; reading on from a recent file",
                            severity="suspect",
                            evidence={"latest": latest},
                        )
                    )
                    await self.rejoin(number + 1, int(latest["segment"]))
                break
            found = await self.take(segment, joining)
            problems += found
            if any(p.severity == "fraud" for p in found):
                break
            joining = False
            number += 1
            self.set_state("segment", number)
            self.set_state("joining", False)
        problems += await self.check_anchor()
        if self.replay_bin and not self.state("broken"):
            problems += await self.replay()
        self.prune()
        for problem in problems:
            self.write_proof(problem)
        return problems

    async def take(self, segment: dict, joining: bool) -> list[Problem]:
        last = int(self.state("last") or 0)
        head = self.state("head")
        if joining or head is None:
            last, head = int(segment["first"]) - 1, segment["prev"]
        try:
            last, head = self.verified(segment, last, head)
        except Broken as broken:
            self.set_state("broken", str(broken))
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
            if entry.get("kind") == "requests":
                listed = (entry.get("input") or {}).get("requests")
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
                    seq=anchor.seq,
                    evidence={"before": seen, "now": anchor.__dict__},
                )
            ]
        self.set_state("anchor", anchor.__dict__)
        wrong = check_anchor(anchor, self.hash_at)
        if wrong:
            self.set_state("broken", wrong)
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
            self.set_state("anchored", max(anchor.seq, self.anchored_through()))
        return []

    async def replay(self) -> list[Problem]:
        """Runs the API's replay over the files read; its state lives beside them, so each run goes on from the last."""
        command = [
            self.replay_bin,
            "--dir",
            str(self.folder),
            "--state",
            str(self.folder / "replay.db"),
            *[part for signer in self.signers.every() for part in ("--signer", signer)],
            "--join",
        ]
        try:
            process = await asyncio.create_subprocess_exec(
                *command, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
            )
            out, err = await asyncio.wait_for(process.communicate(), REPLAY_TIMEOUT_S)
        except (OSError, asyncio.TimeoutError) as error:
            log.warning("the log replay did not run: %r", error)
            return []
        problems = []
        for line in out.decode().splitlines():
            found = json.loads(line)
            if "severity" in found:
                problems.append(
                    Problem(found["what"], found["severity"], found.get("seq"), found)
                )
        if process.returncode not in (0, None) and not problems:
            log.warning("the log replay failed: %s", err.decode()[-2000:])
        return problems

    async def rejoin(self, after: int, latest: int) -> None:
        """Starts again from the first file still there after a gap: the chain is trusted from it until an anchor covers it, and the
        replay restarts at the next snapshot."""
        start = latest
        for number in range(after, latest + 1):
            if await self.get(segment_key(number)) is not None:
                start = number
                break
        for name in ("last", "head"):
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

    def write_proof(self, problem: Problem) -> None:
        if problem.severity != "fraud":
            return
        name = f"{problem.seq or 0:012d}-{int(time.time())}.json"
        proof = {
            "problem": problem.as_dict(),
            "signer": str(self.signers),
            "anchor": self.state("anchor"),
            "storage_url": self.storage_url,
            "files": sorted(p.name for p in (self.folder / "seq").glob("*.json"))[
                -KEEP_FILES:
            ],
        }
        (self.folder / "fraud" / name).write_text(json.dumps(proof, indent=2))
        log.error(
            "task API log: %s (proof in %s)", problem.what, self.folder / "fraud" / name
        )


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
