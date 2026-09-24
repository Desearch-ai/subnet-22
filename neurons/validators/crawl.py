from __future__ import annotations

import asyncio
import contextlib
import logging
import secrets
import time
from collections.abc import Awaitable, Callable

import aiohttp
from yarl import URL

from desearch.client import TaskApiError
from neurons.validators.scoring import (
    MATCH_RATIO,
    MIN_SAMPLES,
    NOT_FETCHED,
    FetchedPage,
    empty_result,
    sample_seed,
)
from neurons.validators.scoring_process import MEMORY_MB, ScoringProcess, Unscorable

PageFetcher = Callable[..., Awaitable[FetchedPage]]

DOWNLOAD_ATTEMPTS = 2
SUBMIT_ATTEMPTS = 3
RETRY_DELAY_S = 2.0
IDLE_DELAY_S = 2.0
MAX_DOWNLOAD_BYTES = 64_000_000
READ_CHUNK = 1 << 20
SCORE_TIMEOUT_S = 300.0
PAUSE_S = 30.0
MAX_PAUSE_S = 600.0
PROVIDER_STREAK = 3

log = logging.getLogger("validator")

# Keeps the sample unpredictable to miners.
SAMPLE_SALT = secrets.token_hex(32)


class DownloadFailed(Exception):
    pass


class UploadMissing(Exception):
    pass


class CrawlValidator:
    def __init__(
        self,
        api,
        fetcher: PageFetcher,
        http: aiohttp.ClientSession,
        min_samples: int = MIN_SAMPLES,
        match_ratio: float = MATCH_RATIO,
        max_download: int = MAX_DOWNLOAD_BYTES,
        score_timeout: float = SCORE_TIMEOUT_S,
        memory_mb: int = MEMORY_MB,
    ):
        self.api = api
        self.fetcher = fetcher
        self.http = http
        self.min_samples = min_samples
        self.match_ratio = match_ratio
        self.max_download = max_download
        self.score_timeout = score_timeout
        self.memory_mb = memory_mb
        self.paused_until = 0.0
        self.pause = PAUSE_S
        self.provider_failures = 0

    async def run(self, stop: asyncio.Event, idle_exit: int = 0) -> None:
        idle = 0
        while not stop.is_set():
            waiting = self.paused_until - time.monotonic()
            if waiting > 0:
                await _sleep(stop, waiting)
                continue
            try:
                result = await self.check_next_task()
            except TaskApiError as exc:
                log.warning("api unavailable: %s", exc)
                await _sleep(stop, RETRY_DELAY_S)
                continue

            if result is not None:
                idle = 0
                continue
            idle += 1
            if idle_exit and idle >= idle_exit:
                return
            await _sleep(stop, IDLE_DELAY_S)

    async def check_next_task(self) -> dict | None:
        job = (await self.api.post("/v1/validation/lease"))["job"]
        if job is None:
            return None

        try:
            result = await self.score_task(job)
        except UploadMissing:
            log.warning("task=%s upload is gone, handing it back", job["task_id"])
            await self.hand_back(job["task_id"], "missing")
            return {}
        except DownloadFailed as exc:
            log.warning("%s, handing it back", exc)
            await self.hand_back(job["task_id"], "download")
            return {}

        if result["verdict"] == "retry":
            self.provider_failures += 1
            log.warning(
                "task=%s ScrapingDog failed %d of %d samples, handing it back",
                job["task_id"],
                result[NOT_FETCHED],
                result["sampled"],
            )
            if self.provider_failures >= PROVIDER_STREAK:
                log.warning(
                    "%d tasks in a row, pausing %.0fs",
                    self.provider_failures,
                    self.pause,
                )
                self.paused_until = time.monotonic() + self.pause
                self.pause = min(self.pause * 2, MAX_PAUSE_S)
            await self.hand_back(job["task_id"], "provider")
            return {}
        self.provider_failures, self.pause = 0, PAUSE_S

        await self.submit_verdict(job["task_id"], result)
        comparable = result["matched"] + result["mismatched"]
        log.info(
            "task=%s miner=%s returned=%d/%d matched=%d/%d verdict=%s reason=%s",
            job["task_id"],
            job["miner"][:10],
            result["returned"],
            len(set(job["urls"])),
            result["matched"],
            comparable,
            result["verdict"],
            result["reason"],
        )
        return result

    async def score_task(self, job: dict) -> dict:
        started = time.monotonic()
        data = await self.download(job)
        session = ScoringProcess(
            data or b"",
            job["urls"],
            sample_seed(job["task_id"], self.api.hotkey, SAMPLE_SALT),
            self.min_samples,
            self.match_ratio,
            self.score_timeout,
            self.memory_mb,
        )
        try:
            await session.start()
            urls = await session.ask("samples")
            pages = await asyncio.gather(*(self._fetch_sample(url) for url in urls))
            needs_rendered_check = await session.ask(
                "doubtful", dict(zip(urls, pages, strict=True))
            )
            rendered = await asyncio.gather(
                *(
                    self._fetch_sample(url, rendered=True)
                    for url in needs_rendered_check
                )
            )
            result = await session.ask(
                "scored", dict(zip(needs_rendered_check, rendered, strict=True))
            )
        except Unscorable as why:
            log.warning("task=%s could not be scored: %s", job["task_id"], why)
            result = empty_result(set(job["urls"]), "unscorable")
        finally:
            await session.aclose()
        return {**result, "took_ms": int((time.monotonic() - started) * 1000)}

    async def download(self, job: dict) -> bytes | None:
        problem = ""
        for attempt in range(DOWNLOAD_ATTEMPTS):
            if attempt:
                await asyncio.sleep(RETRY_DELAY_S)
            try:
                # Presigned: sent as is. Raw: never inflate a Content-Encoding the miner set.
                async with self.http.get(
                    URL(job["download_url"], encoded=True), auto_decompress=False
                ) as response:
                    status = response.status
                    if status == 200:
                        return await self._read_capped(response)
            except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
                problem = type(exc).__name__
                continue
            if status == 404:
                raise UploadMissing(job["task_id"])
            problem = f"HTTP {status}"
            if status < 500:
                break
        raise DownloadFailed(f"task={job['task_id']} download failed: {problem}")

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

    async def _fetch_sample(self, url: str, rendered: bool = False) -> FetchedPage:
        try:
            if rendered:
                return await self.fetcher(url, rendered=True)
            return await self.fetcher(url)
        except Exception as exc:
            return FetchedPage(0, error=f"fetcher_{type(exc).__name__}")


async def _sleep(stop: asyncio.Event, seconds: float) -> None:
    with contextlib.suppress(asyncio.TimeoutError):
        await asyncio.wait_for(stop.wait(), seconds)
