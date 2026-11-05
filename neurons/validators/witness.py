"""This validator's witness records: miners' requests it relayed, what storage held for every completed upload,
its own probes of storage and its reports of storage failing, and the censorship proofs its witnessed requests make
when the task API's anchored log lacks or contradicts the answer the API gave.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import secrets
import sqlite3
import threading
import time
from email.utils import parsedate_to_datetime

import aiohttp
from yarl import URL

from desearch import witness
from desearch.witness import CHECK_AFTER_BLOCKS, LATE_BLOCKS, PROBE, REQUEST, STORAGE

KEEP_S = 7 * 86400
HEAD_TIMEOUT = aiohttp.ClientTimeout(total=20.0)
HEADS_AT_ONCE = 8
PAGE = 500
# A probe writes what a miner's upload looks like: Parquet's magic at both ends, random bytes between, 16 to 256 KB.
PROBE_MAGIC = b"PAR1"
PROBE_MIN_BYTES, PROBE_MAX_BYTES = 16 * 1024, 256 * 1024
PROBE_STEP_TIMEOUT = aiohttp.ClientTimeout(total=30.0)
PROBE_EVERY_S = 60.0
REPORT_TIMEOUT = aiohttp.ClientTimeout(total=30.0)

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
        ref = record["request"] if record["kind"] == REQUEST else record.get("key") or ""
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

    def add_proof(self, proof: dict, ref: str | None = None) -> None:
        with self.lock, self.conn:
            self.conn.execute(
                "INSERT OR REPLACE INTO proofs (request, at, proof) VALUES (?, ?, ?)",
                (ref or proof["request"], time.time(), json.dumps(proof)),
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
        on_failure=None,
    ):
        self.http = http
        self.storage_url = storage_url.rstrip("/")
        self.keypair = keypair
        self.store = store
        self.head = head
        self.at_once = asyncio.Semaphore(HEADS_AT_ONCE)
        self.mismatches = 0
        # Told of every look storage did not answer, or answered with a server error: storage failing this validator.
        self.on_failure = on_failure

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
        if self.on_failure is not None and storage_failed(record):
            self.on_failure(record)
        if record["matches"] is False:
            log.warning(
                f"task={entry.get('task_id')} storage holds another object than its completion pinned:"
                f" HTTP {found['status']}, ETag {found.get('etag')} for {entry.get('etag')}"
            )
        return record


def storage_failed(record: dict) -> bool:
    """A look at storage it did not answer or answered with a server error: storage failing, not the object."""
    if record.get("kind") == PROBE:
        return record.get("ok") is False
    status = record.get("status")
    return status is None or status >= 500


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


class OutcomeLog:
    """What an outcome check asks of the log followed (`JournalFollower`): the anchored entries answering a request, [] once
    the anchored log has listed requests from past `by_block` without one, None while it can't tell."""

    def __init__(self, follower, lag_blocks: int = 5):
        self.follower = follower
        self.lag_blocks = lag_blocks

    def available(self) -> bool:
        """Whether a log is followed at all; without one only arrival can be checked."""
        return self.follower() is not None

    async def outcomes(self, request: str, by_block: int) -> list[dict] | None:
        follower = self.follower()
        if follower is None:
            return None
        found = follower.outcomes(request)
        if found:
            return found
        return [] if follower.covered_block() >= by_block + self.lag_blocks else None


class CensorshipCheck:
    """Asks the log, for every request this validator witnessed long enough ago, whether it holds the outcome the API
    answered with; and, of a request nothing came back for, whether it arrived at all."""

    def __init__(
        self,
        store: WitnessStore,
        head: ChainHead,
        log_view,
        after_blocks: int = CHECK_AFTER_BLOCKS,
        source: str = "",
        outcomes: OutcomeLog | None = None,
        signers=lambda: [],
        late_blocks: int = LATE_BLOCKS,
        keypair=None,
    ):
        self.store = store
        self.head = head
        self.log_view = log_view
        self.after_blocks = after_blocks
        self.source = source
        self.outcomes = outcomes
        # The keys that held the API's log, pinned by this validator: receipts are believed only when they sign them.
        self.signers = signers
        self.late_blocks = late_blocks
        self.keypair = keypair

    async def check(self) -> list[dict]:
        head = self.head.block
        if head is None:
            return []
        found = []
        for row_id, record in self.store.unchecked(head - self.after_blocks + 1):
            proof, settled = await self.judge(record, head)
            if not settled:
                # The log can't tell yet; asked again next pass.
                continue
            self.store.mark_checked(row_id)
            if proof is None:
                continue
            if self.keypair is not None:
                proof = witness.seal(self.keypair, {**proof, "validator": self.keypair.ss58_address})
            self.store.add_proof(proof, record["request"])
            found.append(proof)
            log.error(
                f"Censorship proof ({proof.get('reason', witness.MISSING)}): {record['method']} {record['path']} from"
                f" {record['miner']} witnessed at block {record['block']}: {proof.get('why') or 'not in the log'}"
                f" (checked at block {head})"
            )
        return found

    async def judge(self, record: dict, head: int) -> tuple[dict | None, bool]:
        if self.outcomes is not None and self.outcomes.available() and record.get("version", 1) >= 2:
            proof, settled = await witness.outcome_proof(
                record,
                head,
                self.outcomes.outcomes,
                self.signers(),
                self.after_blocks,
                self.late_blocks,
                self.source,
            )
            if proof is not None or not settled:
                return proof, settled
        if not witness.expects_receipt(record):
            return None, True
        answer = await self.log_view.included(record["request"])
        if answer is None:
            return None, False
        if answer:
            return None, True
        proof = await witness.censorship_proof(
            record, head, _known(answer), self.after_blocks, self.source
        )
        if proof is not None:
            proof["reason"] = witness.MISSING
        return proof, True

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


def signed_post(keypair, path: str, raw: bytes) -> tuple[dict[str, str], dict]:
    """The auth headers of a POST this validator signs, and the request as a witness record names it."""
    digest = hashlib.sha256(raw).hexdigest()
    timestamp, nonce = str(int(time.time())), secrets.token_hex(16)
    signature = keypair.sign(witness.signing_payload("POST", path, digest, timestamp, nonce)).hex()
    headers = {
        "X-Hotkey": keypair.ss58_address,
        "X-Timestamp": timestamp,
        "X-Nonce": nonce,
        "X-Signature": signature,
        "Content-Type": "application/json",
    }
    signed = {
        "miner": keypair.ss58_address,
        "method": "POST",
        "path": path,
        "body_sha256": digest,
        "timestamp": timestamp,
        "nonce": nonce,
        "signature": signature,
    }
    return headers, signed


async def send_witnessed(
    http: aiohttp.ClientSession,
    keypair,
    api_url: str,
    path: str,
    body: dict,
    store: WitnessStore,
    head: ChainHead,
    timeout: aiohttp.ClientTimeout = REPORT_TIMEOUT,
) -> tuple[int | None, bytes]:
    """Sends one signed request to the API and keeps a witness of it signed by this validator, as a relay would of a
    miner's, so the censorship check holds the API to logging it."""
    raw = json.dumps(body, separators=(",", ":")).encode()
    headers, signed = signed_post(keypair, path, raw)
    block, sent_at, started = head.block, time.time(), time.monotonic()
    status, answer, error = None, b"", None
    try:
        async with http.post(f"{api_url.rstrip('/')}{path}", data=raw, headers=headers, timeout=timeout) as response:
            status, answer = response.status, await response.read()
    except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
        error = type(exc).__name__
    answered_ms = int((time.monotonic() - started) * 1000) if status is not None else None
    store.add(witness.request_record(keypair, signed, block, sent_at, status, answer, answered_ms, error))
    return status, answer


def probe_body() -> bytes:
    size = PROBE_MIN_BYTES + secrets.randbelow(PROBE_MAX_BYTES - PROBE_MIN_BYTES)
    return PROBE_MAGIC + os.urandom(size - 2 * len(PROBE_MAGIC)) + PROBE_MAGIC


class StorageProbe:
    """Probes storage the way miners use it, from this validator: through a link the task API hands validators, a
    presigned PUT of an object named and shaped like a miner's upload, a GET of it back and a DELETE. Every probe is a
    signed `probe` record. A failed probe, or a look at a completed upload storage failed, is reported to the API at
    most once a minute; the report is witnessed like a relayed request, so the API's log owes it an entry, and a
    quorum of validators' reports makes a storage outage under the rules whatever the API's own probes say."""

    def __init__(
        self,
        http: aiohttp.ClientSession,
        api_url: str,
        keypair,
        store: WitnessStore,
        head: ChainHead,
        every_s: float = PROBE_EVERY_S,
    ):
        self.http = http
        self.api_url = api_url.rstrip("/")
        self.keypair = keypair
        self.store = store
        self.head = head
        self.every_s = every_s
        self.failed: dict | None = None
        self.reported_at = 0.0

    def noticed(self, record: dict) -> None:
        """A failed look at storage, to report with the next round."""
        self.failed = self.failed or record

    async def link(self) -> dict | None:
        """A probe link, or None when the API gives none: the API's trouble, which says nothing of storage."""
        status, answer = await send_link_request(self.http, self.keypair, self.api_url)
        if status != 200:
            log.warning(f"No storage probe link from the task API: HTTP {status}")
            return None
        return json.loads(answer)

    async def probe(self) -> dict | None:
        link = await self.link()
        if link is None:
            return None
        sent, started = probe_body(), time.monotonic()
        step, detail, doing = None, None, "put"
        try:
            async with self.http.put(
                URL(link["put_url"], encoded=True),
                data=sent,
                headers={"Content-Type": link.get("content_type", "application/vnd.apache.parquet")},
                timeout=PROBE_STEP_TIMEOUT,
            ) as response:
                if response.status >= 300:
                    step, detail = "put", f"HTTP {response.status}"
            if step is None:
                doing = "get"
                async with self.http.get(URL(link["get_url"], encoded=True), timeout=PROBE_STEP_TIMEOUT) as response:
                    read = await response.read()
                    if response.status >= 300:
                        step, detail = "get", f"HTTP {response.status}"
                    elif read != sent:
                        step, detail = "compare", f"read back {len(read)} bytes, wrote {len(sent)}"
        except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
            step, detail = doing, f"{type(exc).__name__}: {exc}"
        took_ms = int((time.monotonic() - started) * 1000)
        if step != "put":
            try:
                async with self.http.delete(URL(link["delete_url"], encoded=True), timeout=PROBE_STEP_TIMEOUT):
                    pass
            except (aiohttp.ClientError, asyncio.TimeoutError):
                pass
        record = witness.sign(
            self.keypair,
            {
                "kind": PROBE,
                "key": link["key"],
                "ok": step is None,
                "step": step,
                "detail": detail,
                "took_ms": took_ms,
                "block": self.head.block,
                "at": time.time(),
            },
        )
        self.store.add(record)
        if step is not None:
            log.warning(f"Storage probe failed at {step} after {took_ms} ms: {detail}")
            self.noticed(record)
        return record

    async def report(self, now: float | None = None) -> int | None:
        """Reports the failure noticed since the last report, at most once a minute; the API's answer status."""
        now = time.time() if now is None else now
        if self.failed is None or now - self.reported_at < 60.0:
            return None
        record, self.failed, self.reported_at = self.failed, None, now
        status, answer = await send_witnessed(
            self.http, self.keypair, self.api_url, "/v1/storage/witness", {"record": record}, self.store, self.head
        )
        if status != 200:
            log.warning(f"The task API did not take this validator's storage witness: HTTP {status} {answer[:200]!r}")
        return status

    async def follow(self, stop: asyncio.Event) -> None:
        while not stop.is_set():
            try:
                await self.probe()
                await self.report()
            except Exception as error:
                log.warning(f"Storage probe failed to run, retrying: {error!r}")
            try:
                await asyncio.wait_for(stop.wait(), self.every_s)
            except asyncio.TimeoutError:
                pass


async def send_link_request(http: aiohttp.ClientSession, keypair, api_url: str) -> tuple[int | None, bytes]:
    path = "/v1/storage/probe-link"
    headers, _ = signed_post(keypair, path, b"{}")
    try:
        async with http.post(f"{api_url}{path}", data=b"{}", headers=headers, timeout=REPORT_TIMEOUT) as response:
            return response.status, await response.read()
    except (aiohttp.ClientError, asyncio.TimeoutError):
        return None, b""


def _known(answer: bool):
    async def included(_: str) -> bool:
        return answer

    return included
