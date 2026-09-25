from __future__ import annotations

import asyncio
import contextlib
import logging
import time

import aiohttp
from yarl import URL

from desearch.client import TaskApiError

DOWNLOAD_ATTEMPTS = 2
SUBMIT_ATTEMPTS = 3
RETRY_DELAY_S = 2.0
IDLE_DELAY_S = 2.0
MAX_DOWNLOAD_BYTES = 64_000_000
READ_CHUNK = 1 << 20
PAUSE_S = 30.0
MAX_PAUSE_S = 600.0
PROVIDER_STREAK = 3

log = logging.getLogger("validator")


class DownloadFailed(Exception):
    pass


class UploadMissing(Exception):
    pass


class TaskChecker:
    """Leases validation jobs of its kinds, judges each one and reports the verdict."""

    kinds: tuple[str, ...] = ()

    def __init__(
        self, api, http: aiohttp.ClientSession, max_download: int = MAX_DOWNLOAD_BYTES
    ):
        self.api = api
        self.http = http
        self.max_download = max_download
        self.paused_until = 0.0
        self.pause = PAUSE_S
        self.provider_failures = 0

    async def run(self, stop: asyncio.Event, idle_exit: int = 0) -> None:
        idle = 0
        while not stop.is_set():
            waiting = self.paused_until - time.monotonic()
            if waiting > 0:
                await sleep_unless_stopped(stop, waiting)
                continue
            try:
                result = await self.check_next_task()
            except TaskApiError as exc:
                log.warning("api unavailable: %s", exc)
                await sleep_unless_stopped(stop, RETRY_DELAY_S)
                continue

            if result is not None:
                idle = 0
                continue
            idle += 1
            if idle_exit and idle >= idle_exit:
                return
            await sleep_unless_stopped(stop, IDLE_DELAY_S)

    async def lease(self) -> dict | None:
        answer = await self.api.post(
            "/v1/validation/lease", {"kinds": list(self.kinds)}
        )
        return answer["job"]

    async def check_next_task(self) -> dict | None:
        raise NotImplementedError

    def provider_failed(self) -> None:
        """Several failures in a row mean our side is down, so back off before trying again."""
        self.provider_failures += 1
        if self.provider_failures >= PROVIDER_STREAK:
            log.warning(
                "%d tasks in a row, pausing %.0fs", self.provider_failures, self.pause
            )
            self.paused_until = time.monotonic() + self.pause
            self.pause = min(self.pause * 2, MAX_PAUSE_S)

    def provider_worked(self) -> None:
        self.provider_failures, self.pause = 0, PAUSE_S

    async def download(self, url: str, task_id: str) -> bytes | None:
        problem = ""
        for attempt in range(DOWNLOAD_ATTEMPTS):
            if attempt:
                await asyncio.sleep(RETRY_DELAY_S)
            try:
                # Presigned: sent as is. Raw: never inflate a Content-Encoding the miner set.
                async with self.http.get(
                    URL(url, encoded=True), auto_decompress=False
                ) as response:
                    status = response.status
                    if status == 200:
                        return await self._read_capped(response)
            except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
                problem = type(exc).__name__
                continue
            if status == 404:
                raise UploadMissing(task_id)
            problem = f"HTTP {status}"
            if status < 500:
                break
        raise DownloadFailed(f"task={task_id} download failed: {problem}")

    async def _read_capped(self, response: aiohttp.ClientResponse) -> bytes | None:
        if (response.content_length or 0) > self.max_download:
            return None
        body = bytearray()
        async for chunk in response.content.iter_chunked(READ_CHUNK):
            body += chunk
            if len(body) > self.max_download:
                return None
        return bytes(body)

    async def hand_back(self, task_id: str, reason: str) -> None:
        try:
            await self.api.post(f"/v1/validation/{task_id}/release", {"reason": reason})
        except TaskApiError as exc:
            if reason == "missing" and exc.status >= 500:
                return await self.hand_back(task_id, "download")
            log.warning("could not hand back task=%s: %s", task_id, exc)

    async def submit_verdict(self, task_id: str, result: dict) -> None:
        for attempt in range(SUBMIT_ATTEMPTS):
            try:
                await self.api.post(f"/v1/validation/{task_id}/score", result)
                return
            except TaskApiError as exc:
                if 0 < exc.status < 500 or attempt == SUBMIT_ATTEMPTS - 1:
                    raise
            await asyncio.sleep(RETRY_DELAY_S)


async def sleep_unless_stopped(stop: asyncio.Event, seconds: float) -> None:
    with contextlib.suppress(asyncio.TimeoutError):
        await asyncio.wait_for(stop.wait(), seconds)
