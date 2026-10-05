from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from collections import deque

import aiohttp
from yarl import URL

from desearch.client import TaskApiError
from desearch.credit import crawl_credit, embed_credit
from desearch.kinds import CRAWL, EMBED
from desearch.manifest import OPEN_LIST_KEY
from desearch.manifest import verify as verify_manifest

LIST_TIMEOUT = aiohttp.ClientTimeout(total=30.0)
LIST_CACHE_S = 2.0
REPORTED_KEEP_S = 3600.0
DOWNLOAD_ATTEMPTS = 2
SUBMIT_ATTEMPTS = 3
RETRY_DELAY_S = 2.0
IDLE_DELAY_S = 2.0
MAX_DOWNLOAD_BYTES = 100_000_000
READ_CHUNK = 1 << 20
PAUSE_S = 30.0
MAX_PAUSE_S = 600.0
PROVIDER_STREAK = 3
FAILURE_STREAK = 3
RECENT_CHECKS = 20
HEALTHY_AFTER = 5
HEALTHY_SHARE = 0.8
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
        storage_url: str = "",
        seeds=None,
        signer: str = "",
    ):
        self.api = api
        self.http = http
        self.max_download = max_download
        self.ledger = ledger
        self.storage_url = storage_url.rstrip("/")
        self.seeds = seeds
        self.signer = signer
        self.paused_until = 0.0
        self.pause = PAUSE_S
        self.provider_failures = 0
        self.failures = 0
        self.refused: str | None = None
        self.in_flight: set[str] = set()
        self.recent: deque[bool] = deque(maxlen=RECENT_CHECKS)
        self.deferred: dict[str, float] = {}
        self.reported: dict[str, float] = {}
        self.listed: list[dict] = []
        self.listed_at = float("-inf")

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
        self.recent.append(False)

    def scoring_worked(self) -> None:
        self.failures = 0
        self.recent.append(True)

    def healthy(self) -> bool:
        """This checker scores most uploads, so one it cannot score says something about the upload."""
        return len(self.recent) >= HEALTHY_AFTER and sum(
            self.recent
        ) >= HEALTHY_SHARE * len(self.recent)

    async def run(self, stop: asyncio.Event, idle_exit: int = 0) -> None:
        idle = 0
        while not stop.is_set():
            waiting = self.paused_until - time.monotonic()
            if waiting > 0:
                await sleep_unless_stopped(stop, waiting)
                continue
            job = await self.next_job()
            if job is None:
                idle += 1
                if idle_exit and idle >= idle_exit:
                    return
                await sleep_unless_stopped(stop, IDLE_DELAY_S)
                continue
            idle = 0
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

    async def open_list(self) -> list[dict]:
        """The open uploads as the task API last listed them, one read shared by every loop."""
        if time.monotonic() - self.listed_at >= LIST_CACHE_S:
            self.listed = await self.fetch_open_list()
            self.listed_at = time.monotonic()
        return list(self.listed)

    async def fetch_open_list(self) -> list[dict]:
        url = f"{self.storage_url}/{OPEN_LIST_KEY}"
        async with self.http.get(URL(url, encoded=True), timeout=LIST_TIMEOUT) as r:
            if r.status != 200:
                raise DownloadFailed(f"open list: HTTP {r.status}")
            listing = await r.json(content_type=None)
        return list(listing.get("uploads", []))

    async def next_job(self) -> dict | None:
        """The oldest listed upload of our kinds whose seed exists and nobody here is on; it is ours until released."""
        now = time.monotonic()
        self.deferred = {t: until for t, until in self.deferred.items() if until > now}
        keep = time.time() - REPORTED_KEEP_S
        self.reported = {t: at for t, at in self.reported.items() if at > keep}
        try:
            uploads = await self.open_list()
        except (
            aiohttp.ClientError,
            asyncio.TimeoutError,
            DownloadFailed,
            ValueError,
        ) as exc:
            log.warning("open list unavailable: %r", exc)
            return None
        waiting = [
            manifest
            for manifest in uploads
            if manifest.get("kind", CRAWL) in self.kinds
            and manifest.get("task_id")
            and manifest["task_id"] not in self.in_flight
            and manifest["task_id"] not in self.deferred
            and manifest.get("key") not in self.reported
        ]
        for manifest in self.oldest_first(waiting):
            task_id = manifest["task_id"]
            if task_id in self.in_flight:
                continue
            if not verify_manifest(manifest, self.signer):
                log.warning(
                    "task=%s is not signed by %s, skipped", task_id, self.signer
                )
                continue
            # Taken before the seed lookup, so loops waiting on it never pick the same upload.
            self.in_flight.add(task_id)
            try:
                seed = await self.seeds(manifest["seed_block"])
            except Exception as exc:
                self.in_flight.discard(task_id)
                log.warning("task=%s seed block unavailable: %r", task_id, exc)
                return None
            if seed is not None:
                return self.job_of(manifest, seed)
            self.in_flight.discard(task_id)
        return None

    def oldest_first(self, waiting: list[dict]) -> list[dict]:
        return sorted(waiting, key=lambda manifest: manifest.get("completed_at", 0.0))

    def job_of(self, manifest: dict, seed: str) -> dict:
        job = {
            **manifest,
            "seed": seed,
            "download_url": f"{self.storage_url}/{manifest['key']}",
        }
        if manifest.get("input_key"):
            job["input"] = {
                "url": f"{self.storage_url}/{manifest['input_key']}",
                "sha256": manifest.get("input_sha256", ""),
            }
        return job

    def defer(self, task_id: str, seconds: float = DEFER_S) -> None:
        self.deferred[task_id] = time.monotonic() + seconds

    async def check(self, job: dict) -> dict | None:
        raise NotImplementedError

    def note_verdict(self, job: dict, result: dict) -> None:
        """What this validator itself decided, kept for its own weights."""
        if self.ledger is None:
            return
        kind = job.get("kind", CRAWL)
        if kind == EMBED:
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
        if kind == CRAWL:
            returned = result.get("returned", 0)
            self.ledger.record_check(
                job["key"],
                job["miner"],
                job["completed_at"],
                result["verdict"],
                credited,
                returned,
                returned - result.get("error_rows", 0),
                assigned,
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

    async def submit_verdict(self, job: dict, result: dict) -> bool:
        """False when the upload finalized without this verdict; it is still ours to keep."""
        task_id = job["task_id"]
        for attempt in range(SUBMIT_ATTEMPTS):
            try:
                await self.api.post(f"/v1/validation/{task_id}/score", result)
                self.refused = None
                self.reported[job["key"]] = time.time()
                return True
            except TaskApiError as exc:
                if exc.status == 409:
                    log.info(
                        "task=%s finalized before our verdict: %s", task_id, exc.detail
                    )
                    self.reported[job["key"]] = time.time()
                    return False
                if 0 < exc.status < 500 or attempt == SUBMIT_ATTEMPTS - 1:
                    raise
            await asyncio.sleep(RETRY_DELAY_S)
        return False


async def sleep_unless_stopped(stop: asyncio.Event, seconds: float) -> None:
    with contextlib.suppress(asyncio.TimeoutError):
        await asyncio.wait_for(stop.wait(), seconds)
