"""Every miner's completed uploads, read from the numbered files the task API signs into the uploads bucket."""

from __future__ import annotations

import logging
import time

import aiohttp

from desearch.credit import SHARE_WINDOW_H
from desearch.manifest import UPLOAD_LOG_LATEST, upload_log_key, verify_log

NEXT_SEQ = "upload_log_next"
# A new validator reads back at most this many files, a day at one every 30 seconds.
BACKFILL_FILES = 3000
READ_TIMEOUT = aiohttp.ClientTimeout(total=30.0)

log = logging.getLogger("validator")


class UploadLog:
    def __init__(
        self, http: aiohttp.ClientSession, storage_url: str, signer: str, ledger
    ):
        self.http = http
        self.storage_url = storage_url.rstrip("/")
        self.signer = signer
        self.ledger = ledger

    async def poll(self) -> int:
        """Reads every file written since the last poll; returns how many uploads were new."""
        latest = await self.latest()
        if latest is None:
            return 0
        cursor = self.ledger.state(NEXT_SEQ)
        if cursor is None:
            added = await self.backfill(latest)
            self.ledger.set_state(NEXT_SEQ, str(latest + 1))
            return added
        added = 0
        for seq in range(int(cursor), latest + 1):
            found = await self.read(seq)
            if found is None:
                break
            added += self.ledger.add_uploads(found["entries"])
            self.ledger.set_state(NEXT_SEQ, str(seq + 1))
        return added

    async def backfill(self, latest: int) -> int:
        """Back far enough to cover the scoring window."""
        since = time.time() - SHARE_WINDOW_H * 3600
        added = 0
        for seq in range(latest, max(0, latest - BACKFILL_FILES), -1):
            found = await self.read(seq)
            if found is None:
                continue
            added += self.ledger.add_uploads(found["entries"])
            if found["written_at"] is not None and found["written_at"] < since:
                break
        return added

    async def latest(self) -> int | None:
        found = await self.get(UPLOAD_LOG_LATEST)
        return int(found["seq"]) if found else None

    async def read(self, seq: int) -> dict | None:
        """The file, or None while it is not there; one not signed by the task API holds nothing."""
        found = await self.get(upload_log_key(seq))
        if found is None:
            return None
        if not verify_log(found, self.signer):
            log.warning("upload log %d is not signed by %s, skipped", seq, self.signer)
            return {"entries": [], "written_at": None}
        return found

    async def get(self, key: str) -> dict | None:
        async with self.http.get(
            f"{self.storage_url}/{key}", timeout=READ_TIMEOUT
        ) as response:
            if response.status == 404:
                return None
            response.raise_for_status()
            return await response.json(content_type=None)
