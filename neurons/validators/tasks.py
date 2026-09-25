from __future__ import annotations

import asyncio
import contextlib
import logging
import time

import aiohttp
from yarl import URL

from desearch.client import TaskApiError
from desearch.credit import crawl_credit, embed_credit

OPEN_PATH = "/v1/validation/open"
DOWNLOAD_ATTEMPTS = 2
SUBMIT_ATTEMPTS = 3
RETRY_DELAY_S = 2.0
IDLE_DELAY_S = 2.0
MAX_DOWNLOAD_BYTES = 64_000_000
READ_CHUNK = 1 << 20
PAUSE_S = 30.0
MAX_PAUSE_S = 600.0
PROVIDER_STREAK = 3
FAILURE_STREAK = 3
DEFER_S = 60.0

log = logging.getLogger("validator")


class DownloadFailed(Exception):
    pass


class UploadMissing(Exception):
    pass


class TaskChecker:
    """Walks the open uploads of its kinds, judges each one and reports the verdict."""

    kinds: tuple[str, ...] = ()

    def __init__(
        self,
        api,
        http: aiohttp.ClientSession,
        max_download: int = MAX_DOWNLOAD_BYTES,
        ledger=None,
    ):
        self.api = api
        self.http = http
        self.max_download = max_download
        self.ledger = ledger
        self.paused_until = 0.0
        self.pause = PAUSE_S
        self.provider_failures = 0
        self.failures = 0
        self.refused: str | None = None
        self.in_flight: set[str] = set()
        self.deferred: dict[str, float] = {}

    @property
    def trouble(self) -> str | None:
        """Why this checker cannot vouch for its verdicts right now, if it cannot."""
        if self.refused:
            return f"the task API refuses this validator: {self.refused}"
        if self.provider_failures >= PROVIDER_STREAK:
            return f"the provider failed on {self.provider_failures} tasks in a row"
        if self.failures >= FAILURE_STREAK:
            return f"{self.failures} tasks in a row could not be scored"
        return None

    def note_refusal(self, exc: TaskApiError) -> None:
        """A 403 means the API no longer takes this validator's word, not that it is down."""
        if exc.status == 403:
            self.refused = exc.detail

    def scoring_failed(self) -> None:
        self.failures += 1

    def scoring_worked(self) -> None:
        self.failures = 0

    async def run(self, stop: asyncio.Event, idle_exit: int = 0) -> None:
        idle = 0
        while not stop.is_set():
            waiting = self.paused_until - time.monotonic()
            if waiting > 0:
                await sleep_unless_stopped(stop, waiting)
                continue
            try:
                job = await self.next_job()
            except TaskApiError as exc:
                self.note_refusal(exc)
                log.warning("api unavailable: %s", exc)
                await sleep_unless_stopped(stop, RETRY_DELAY_S)
                continue
            self.refused = None

            if job is None:
                idle += 1
                if idle_exit and idle >= idle_exit:
                    return
                await sleep_unless_stopped(stop, IDLE_DELAY_S)
                continue
            idle = 0
            self.in_flight.add(job["task_id"])
            try:
                await self.check(job)
            except TaskApiError as exc:
                self.note_refusal(exc)
                self.defer(job["task_id"])
                log.warning("task=%s not recorded: %s", job["task_id"], exc)
            except Exception:
                self.scoring_failed()
                self.defer(job["task_id"])
                log.exception("task=%s could not be checked", job["task_id"])
            finally:
                self.in_flight.discard(job["task_id"])

    async def next_job(self) -> dict | None:
        """The oldest open upload nobody in this process is on, or None."""
        now = time.monotonic()
        self.deferred = {t: until for t, until in self.deferred.items() if until > now}
        skip = [*self.in_flight, *self.deferred][:64]
        answer = await self.api.post(
            OPEN_PATH, {"kinds": list(self.kinds), "skip": skip}
        )
        for job in answer.get("jobs", []):
            if job["task_id"] not in self.in_flight:
                return job
        return None

    def defer(self, task_id: str, seconds: float = DEFER_S) -> None:
        self.deferred[task_id] = time.monotonic() + seconds

    async def check(self, job: dict) -> dict | None:
        raise NotImplementedError

    def note_verdict(self, job: dict, result: dict) -> None:
        """What this validator itself decided, kept for its own weights."""
        if self.ledger is None:
            return
        kind = job.get("kind", "crawl")
        if kind == "embed":
            credited, assigned = embed_credit(job, result), job.get("texts", 0)
        else:
            credited, assigned = crawl_credit(result), len(set(job["urls"]))
        self.ledger.record(
            job["task_id"],
            kind,
            job["miner"],
            result["verdict"],
            credited,
            assigned,
            result.get("returned", 0),
        )

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
        """Only a lost upload is the API's business; anything else is tried again later."""
        self.defer(task_id)
        if reason != "missing":
            return
        try:
            await self.api.post(f"/v1/validation/{task_id}/release", {"reason": reason})
        except TaskApiError as exc:
            log.warning("could not report the missing upload task=%s: %s", task_id, exc)

    async def submit_verdict(self, task_id: str, result: dict) -> bool:
        """False when the upload finalized without this verdict, which is not a fault."""
        for attempt in range(SUBMIT_ATTEMPTS):
            try:
                await self.api.post(f"/v1/validation/{task_id}/score", result)
                return True
            except TaskApiError as exc:
                if exc.status == 409:
                    log.info(
                        "task=%s finalized before our verdict: %s", task_id, exc.detail
                    )
                    return False
                if 0 < exc.status < 500 or attempt == SUBMIT_ATTEMPTS - 1:
                    raise
            await asyncio.sleep(RETRY_DELAY_S)
        return False


async def sleep_unless_stopped(stop: asyncio.Event, seconds: float) -> None:
    with contextlib.suppress(asyncio.TimeoutError):
        await asyncio.wait_for(stop.wait(), seconds)
