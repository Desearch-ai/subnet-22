"""Every miner's completed uploads, read from the numbered files the task API signs into the uploads bucket.

The files form a chain: each names the SHA-256 of the previous file's bytes (`prev`) and when that file was written
(`prev_written_at`). A file whose link is not the file read before it is a fork; a file that is gone while a later one
names it, younger than the bucket keeps files, was withheld. Both are fraud the API signed for, and are written down as
proofs rather than read past as empty.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time

import aiohttp

from desearch.credit import SHARE_WINDOW_H
from desearch.manifest import UPLOAD_LOG_LATEST, upload_log_key, verify_log
from neurons.validators.custody import as_signers

NEXT_SEQ = "upload_log_next"
# The last file read: {"seq", "hash", "written_at"}, the link the next file must name.
HEAD = "upload_log_head"
# A file found missing below the newest: {"seq", "since"}, so it is judged only once it stayed missing a while.
MISSING = "upload_log_missing"
# A new validator reads back at most this many files, a day at one every 30 seconds.
BACKFILL_FILES = 3000
READ_TIMEOUT = aiohttp.ClientTimeout(total=30.0)
# The bucket keeps log files a day; one younger than that is there unless someone took it away.
RETENTION_S = 86400.0
# How long a file must stay missing before it is judged, so a slow store is not taken for a withheld file.
MISSING_GRACE_S = 120.0
UPLOAD_LOG = "upload_log"

log = logging.getLogger("validator")


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class UploadLog:
    def __init__(
        self,
        http: aiohttp.ClientSession,
        storage_url: str,
        signer,
        ledger,
        observer=None,
        on_problem=None,
        retention_s: float = RETENTION_S,
        grace_s: float = MISSING_GRACE_S,
    ):
        self.http = http
        self.storage_url = storage_url.rstrip("/")
        # One address, or the custody chain naming who signed which file.
        self.signers = as_signers(signer)
        self.ledger = ledger
        # Looks at every new upload in storage and signs what it saw; backfilled ones may be gone already.
        self.observer = observer
        # Told of each fork or withheld file as a dict: {"what", "severity", "seq", "evidence"}.
        self.on_problem = on_problem
        self.retention_s = retention_s
        self.grace_s = grace_s
        self.problems: list[dict] = []

    def state(self, name: str):
        found = self.ledger.state(name)
        return json.loads(found) if found else None

    def set_state(self, name: str, value) -> None:
        self.ledger.set_state(name, json.dumps(value))

    def problem(self, what: str, seq: int, evidence: dict, severity: str = "fraud") -> None:
        found = {"kind": UPLOAD_LOG, "what": what, "severity": severity, "seq": seq, "evidence": evidence}
        self.problems.append(found)
        (log.error if severity == "fraud" else log.warning)(f"upload log: {what}")
        if self.on_problem is not None:
            self.on_problem(found)

    async def poll(self) -> int:
        """Reads every file written since the last poll; returns how many uploads were new."""
        latest = await self.latest()
        if latest is None:
            return 0
        newest = int(latest["seq"])
        cursor = self.ledger.state(NEXT_SEQ)
        if cursor is None:
            added = await self.backfill(newest)
            self.ledger.set_state(NEXT_SEQ, str(newest + 1))
            return added
        added = 0
        seq = int(cursor)
        while seq <= newest:
            found, raw = await self.read(seq)
            if found is None:
                if seq == newest:
                    break
                after = await self.gone(seq, newest)
                if after is None:
                    break
                # Files after a gap are read on; the uploads the missing ones held are lost to this count.
                self.ledger.set_state(NEXT_SEQ, str(after))
                seq = after
                continue
            if raw is not None:
                self.link(seq, found, raw, latest)
            added += self.ledger.add_uploads(found["entries"])
            self.ledger.set_state(NEXT_SEQ, str(seq + 1))
            await self.observe(found["entries"])
            seq += 1
        return added

    def link(self, seq: int, found: dict, raw: bytes, latest: dict) -> None:
        """Checks the file names the one read before it, and keeps its hash for the next."""
        digest = sha256(raw)
        head = self.state(HEAD)
        if "prev" in found and head and head.get("seq") == seq - 1 and found["prev"] != head["hash"]:
            self.problem(
                f"file {seq} follows {found['prev']}, not file {seq - 1} as read ({head['hash']})",
                seq,
                {"file": found, "read_before": head},
            )
        if latest.get("head") and int(latest["seq"]) == seq and latest["head"] != digest:
            self.problem(
                f"latest.json names {latest['head']} as file {seq}, the file read hashes {digest}",
                seq,
                {"file": found, "latest": latest},
            )
        self.set_state(HEAD, {"seq": seq, "hash": digest, "written_at": found.get("written_at")})

    async def gone(self, seq: int, newest: int, now: float | None = None) -> int | None:
        """A file missing below the newest. None while it has not been missing long enough to judge; then the number to
        read on from, after a fraud proof when the next file shows the missing ones were younger than the bucket keeps."""
        now = time.time() if now is None else now
        missing = self.state(MISSING)
        if not missing or missing.get("seq") != seq:
            self.set_state(MISSING, {"seq": seq, "since": now})
            return None
        if now - float(missing["since"]) < self.grace_s:
            return None
        after, found = seq + 1, None
        while after <= newest:
            found, _ = await self.read(after)
            if found is not None:
                break
            after += 1
        last_written = (found or {}).get("prev_written_at")
        evidence = {"missing": list(range(seq, after)), "next": found, "checked_at": now, "retention_s": self.retention_s}
        if found is not None and last_written is not None and float(last_written) >= now - self.retention_s:
            self.problem(
                f"files {seq}..{after - 1} are gone while file {after} names them, written"
                f" {int(now - float(last_written))} s ago: withheld",
                seq,
                evidence,
            )
        else:
            self.problem(f"files {seq}..{after - 1} are gone while {newest} exists; read on", seq, evidence, "suspect")
        self.set_state(MISSING, None)
        return after

    async def observe(self, entries: list[dict]) -> None:
        if self.observer is None or not entries:
            return
        try:
            await self.observer.observe(entries)
        except Exception as error:
            log.warning(
                "could not look at %d uploads in storage: %r", len(entries), error
            )

    async def backfill(self, latest: int, now: float | None = None) -> int:
        """Back far enough to cover the scoring window, checking each file is the one the file after it names."""
        now = time.time() if now is None else now
        since = now - SHARE_WINDOW_H * 3600
        added = 0
        after: dict | None = None
        for seq in range(latest, max(0, latest - BACKFILL_FILES), -1):
            found, raw = await self.read(seq)
            if found is None:
                last_written = (after or {}).get("prev_written_at")
                if last_written is not None and float(last_written) >= now - self.retention_s:
                    self.problem(
                        f"file {seq} is gone while file {seq + 1} names it, written {int(now - float(last_written))} s ago: withheld",
                        seq,
                        {"missing": [seq], "next": after, "checked_at": now, "retention_s": self.retention_s},
                    )
                after = None
                continue
            if raw is not None:
                if after is not None and "prev" in after and after["prev"] != sha256(raw):
                    self.problem(
                        f"file {seq + 1} follows {after['prev']}, not file {seq} ({sha256(raw)})",
                        seq + 1,
                        {"file": after, "before": found},
                    )
                if seq == latest:
                    self.set_state(HEAD, {"seq": seq, "hash": sha256(raw), "written_at": found.get("written_at")})
            added += self.ledger.add_uploads(found["entries"])
            after = found if raw is not None else None
            if found["written_at"] is not None and found["written_at"] < since:
                break
        return added

    async def latest(self) -> dict | None:
        """The newest file's number, and its hash when the API signed it; one signed by nobody it may sign as is ignored."""
        raw = await self.get(UPLOAD_LOG_LATEST)
        if raw is None:
            return None
        found = json.loads(raw)
        if "signature" in found:
            allowed = self.signers.for_log(int(found["seq"]))
            if not any(verify_log(found, signer) for signer in allowed):
                log.warning("upload log latest.json is not signed by %s, ignored", " or ".join(allowed) or "a followed signer")
                return None
        return found

    async def read(self, seq: int) -> tuple[dict | None, bytes | None]:
        """The file and its bytes, or (None, None) while it is not there; one not signed by the task API holds nothing."""
        raw = await self.get(upload_log_key(seq))
        if raw is None:
            return None, None
        found = json.loads(raw)
        allowed = self.signers.for_log(seq)
        if not any(verify_log(found, signer) for signer in allowed):
            log.warning(
                "upload log %d is not signed by %s, skipped",
                seq,
                " or ".join(allowed) or "a followed signer",
            )
            return {"entries": [], "written_at": None}, None
        return found, raw

    async def get(self, key: str) -> bytes | None:
        async with self.http.get(
            f"{self.storage_url}/{key}", timeout=READ_TIMEOUT
        ) as response:
            # A store that keeps its bucket private answers 403, not 404, for an object it does not have.
            if response.status in (403, 404):
                return None
            response.raise_for_status()
            return await response.read()
