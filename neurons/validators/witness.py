"""This validator's witness records: miners' requests it relayed, what storage held for every completed upload,
and the censorship proofs its relayed requests make when the task API's log lacks them.
"""

from __future__ import annotations

import asyncio
import json
import logging
import sqlite3
import threading
import time
from email.utils import parsedate_to_datetime

import aiohttp
from yarl import URL

from desearch import witness
from desearch.witness import CHECK_AFTER_BLOCKS, REQUEST, STORAGE

KEEP_S = 7 * 86400
HEAD_TIMEOUT = aiohttp.ClientTimeout(total=20.0)
HEADS_AT_ONCE = 8
PAGE = 500

log = logging.getLogger("validator")


class WitnessStore:
    """Signed records kept in SQLite, readable by anyone from the relay's endpoints."""

    def __init__(self, path: str):
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self.lock = threading.Lock()
        with self.lock, self.conn:
            self.conn.executescript("""
                CREATE TABLE IF NOT EXISTS records (
                    id      INTEGER PRIMARY KEY AUTOINCREMENT,
                    kind    TEXT NOT NULL,
                    ref     TEXT NOT NULL,
                    miner   TEXT,
                    block   INTEGER,
                    at      REAL NOT NULL,
                    checked INTEGER NOT NULL DEFAULT 0,
                    record  TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS records_kind ON records (kind, id);
                CREATE INDEX IF NOT EXISTS records_ref ON records (ref);
                CREATE TABLE IF NOT EXISTS proofs (
                    request TEXT PRIMARY KEY,
                    at      REAL NOT NULL,
                    proof   TEXT NOT NULL
                );
                """)

    def add(self, record: dict) -> None:
        ref = record["request"] if record["kind"] == REQUEST else record["key"]
        at = record.get("forwarded_at") or record.get("at") or time.time()
        with self.lock, self.conn:
            self.conn.execute(
                "INSERT INTO records (kind, ref, miner, block, at, record) VALUES (?, ?, ?, ?, ?, ?)",
                (
                    record["kind"],
                    ref,
                    record.get("miner"),
                    record.get("block"),
                    at,
                    json.dumps(record),
                ),
            )

    def records(
        self,
        kind: str,
        before_block: int | None = None,
        after_id: int = 0,
        limit: int = PAGE,
        miner: str | None = None,
    ) -> list[dict]:
        """Records of a kind with their row id as `id`, oldest first, only those forwarded before `before_block`."""
        sql = "SELECT id, record FROM records WHERE kind = ? AND id > ?"
        args: list = [kind, after_id]
        if before_block is not None:
            sql += " AND block IS NOT NULL AND block < ?"
            args.append(before_block)
        if miner:
            sql += " AND miner = ?"
            args.append(miner)
        sql += " ORDER BY id LIMIT ?"
        args.append(max(1, min(limit, PAGE)))
        with self.lock:
            rows = self.conn.execute(sql, args).fetchall()
        return [{"id": row_id, **json.loads(record)} for row_id, record in rows]

    def by_ref(self, ref: str) -> list[dict]:
        with self.lock:
            rows = self.conn.execute(
                "SELECT record FROM records WHERE ref = ? ORDER BY id", (ref,)
            ).fetchall()
        return [json.loads(record) for (record,) in rows]

    def unchecked(self, before_block: int, limit: int = PAGE) -> list[tuple[int, dict]]:
        with self.lock:
            rows = self.conn.execute(
                "SELECT id, record FROM records WHERE kind = ? AND checked = 0 AND block < ? ORDER BY id LIMIT ?",
                (REQUEST, before_block, limit),
            ).fetchall()
        return [(row_id, json.loads(record)) for row_id, record in rows]

    def mark_checked(self, row_id: int) -> None:
        with self.lock, self.conn:
            self.conn.execute("UPDATE records SET checked = 1 WHERE id = ?", (row_id,))

    def add_proof(self, proof: dict) -> None:
        with self.lock, self.conn:
            self.conn.execute(
                "INSERT OR REPLACE INTO proofs (request, at, proof) VALUES (?, ?, ?)",
                (proof["request"], time.time(), json.dumps(proof)),
            )

    def proofs(self, since: float = 0.0, limit: int = PAGE) -> list[dict]:
        with self.lock:
            rows = self.conn.execute(
                "SELECT proof FROM proofs WHERE at >= ? ORDER BY at LIMIT ?",
                (since, max(1, min(limit, PAGE))),
            ).fetchall()
        return [json.loads(proof) for (proof,) in rows]

    def prune(self, now: float | None = None) -> None:
        cutoff = (now or time.time()) - KEEP_S
        with self.lock, self.conn:
            self.conn.execute("DELETE FROM records WHERE at < ?", (cutoff,))
            self.conn.execute("DELETE FROM proofs WHERE at < ?", (cutoff,))

    def close(self) -> None:
        self.conn.close()


class ChainHead:
    """The newest block this validator's own node reports, refreshed in the background so a relayed request is
    never held up on a chain call."""

    def __init__(self, read, every_s: float = 6.0):
        self.read = read
        self.every_s = every_s
        self.block: int | None = None

    async def refresh(self) -> int | None:
        try:
            self.block = int(await self.read())
        except Exception as error:
            log.warning(
                f"Chain head unavailable, keeping block {self.block}: {error!r}"
            )
        return self.block

    async def follow(self, stop: asyncio.Event) -> None:
        while not stop.is_set():
            await self.refresh()
            try:
                await asyncio.wait_for(stop.wait(), self.every_s)
            except asyncio.TimeoutError:
                pass


def lock_until(value: str | None) -> float | None:
    if not value:
        return None
    try:
        return parsedate_to_datetime(value).timestamp()
    except (TypeError, ValueError):
        try:
            from datetime import datetime

            return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
        except ValueError:
            return None


def same_etag(a: str | None, b: str | None) -> bool:
    return bool(a) and bool(b) and a.strip().strip('"') == b.strip().strip('"')


class StorageWitness:
    """Looks at every completed upload the upload log lists, through the public bucket as anyone would, and signs
    what it saw: whether the version the completion pinned is there, unchanged, and under lock.
    """

    def __init__(
        self,
        http: aiohttp.ClientSession,
        storage_url: str,
        keypair,
        store: WitnessStore,
        head: ChainHead,
    ):
        self.http = http
        self.storage_url = storage_url.rstrip("/")
        self.keypair = keypair
        self.store = store
        self.head = head
        self.at_once = asyncio.Semaphore(HEADS_AT_ONCE)
        self.mismatches = 0

    async def observe(self, entries: list[dict]) -> int:
        """Signs one observation per entry; returns how many found something other than what was pinned."""
        seen = await asyncio.gather(*(self.observe_one(entry) for entry in entries))
        changed = sum(1 for record in seen if record and record["matches"] is False)
        self.mismatches += changed
        return changed

    async def observe_one(self, entry: dict) -> dict | None:
        key = entry.get("key")
        if not key:
            return None
        pinned_version = entry.get("version")
        url = f"{self.storage_url}/{key}"
        if pinned_version:
            url += f"?versionId={pinned_version}"
        found = {"status": None, "error": None}
        async with self.at_once:
            try:
                async with self.http.head(
                    URL(url, encoded=True), timeout=HEAD_TIMEOUT
                ) as response:
                    headers = response.headers
                    found = {
                        "status": response.status,
                        "etag": headers.get("ETag"),
                        "object_version": headers.get("x-amz-version-id"),
                        "size": (
                            int(headers["Content-Length"])
                            if headers.get("Content-Length", "").isdigit()
                            else None
                        ),
                        "last_modified": headers.get("Last-Modified"),
                        "lock_mode": headers.get("x-amz-object-lock-mode"),
                        "lock_until": lock_until(
                            headers.get("x-amz-object-lock-retain-until-date")
                        ),
                        "error": None,
                    }
            except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
                found["error"] = type(exc).__name__
        record = witness.sign(
            self.keypair,
            {
                "kind": STORAGE,
                "task_id": entry.get("task_id"),
                "miner": entry.get("miner"),
                "key": key,
                "pinned_etag": entry.get("etag"),
                "pinned_version": pinned_version,
                **found,
                "matches": matches(entry, found),
                "block": self.head.block,
                "at": time.time(),
            },
        )
        self.store.add(record)
        if record["matches"] is False:
            log.warning(
                f"task={entry.get('task_id')} storage holds another object than its completion pinned:"
                f" HTTP {found['status']}, ETag {found.get('etag')} for {entry.get('etag')}"
            )
        return record


def matches(entry: dict, found: dict) -> bool | None:
    """True when storage holds what the completion pinned, False when it holds something else or nothing, None
    when the log named nothing to compare or storage did not answer."""
    status = found.get("status")
    if status is None or status >= 500 or status == 403:
        return None
    if status == 404:
        return False
    if entry.get("version"):
        return found.get("object_version") in (None, entry["version"]) and (
            not entry.get("etag") or same_etag(found.get("etag"), entry["etag"])
        )
    if entry.get("etag"):
        return same_etag(found.get("etag"), entry["etag"])
    return None


class CensorshipCheck:
    """Asks the log, for every request this validator relayed long enough ago, whether it holds it."""

    def __init__(
        self,
        store: WitnessStore,
        head: ChainHead,
        log_view,
        after_blocks: int = CHECK_AFTER_BLOCKS,
        source: str = "",
    ):
        self.store = store
        self.head = head
        self.log_view = log_view
        self.after_blocks = after_blocks
        self.source = source

    async def check(self) -> list[dict]:
        head = self.head.block
        if head is None:
            return []
        found = []
        for row_id, record in self.store.unchecked(head - self.after_blocks + 1):
            answer = await self.log_view.included(record["request"])
            if answer is None:
                # The log can't tell yet; asked again next pass.
                continue
            self.store.mark_checked(row_id)
            if answer:
                continue
            proof = await witness.censorship_proof(
                record, head, _known(answer), self.after_blocks, self.source
            )
            if proof:
                self.store.add_proof(proof)
                found.append(proof)
                log.error(
                    f"Censorship proof: {record['method']} {record['path']} from miner {record['miner']} relayed at"
                    f" block {record['block']} is not in the task API's log at block {head}"
                )
        return found

    async def follow(self, stop: asyncio.Event, every_s: float = 60.0) -> None:
        while not stop.is_set():
            try:
                await self.check()
            except Exception as error:
                log.warning(f"Witness check failed, retrying: {error!r}")
            try:
                await asyncio.wait_for(stop.wait(), every_s)
            except asyncio.TimeoutError:
                pass


def _known(answer: bool):
    async def included(_: str) -> bool:
        return answer

    return included
