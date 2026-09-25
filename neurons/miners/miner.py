from __future__ import annotations

import asyncio
import json
import logging
import signal
import time
from concurrent.futures import ThreadPoolExecutor

import aiohttp
from yarl import URL

from desearch import env
from desearch.client import TaskApiClient, TaskApiError
from desearch.fetch import Fetched, Fetcher, ScrapingDog, needs_fallback
from neurons.miners.config import ENV_FILE, Settings
from neurons.miners.rows import UploadWriter, build_row, error_row

log = logging.getLogger("miner")

BACKOFF = {"QUEUE_EMPTY": 2.0, "NO_CAPACITY": 1.0}
ERROR_BACKOFF = 2.0
MAX_BACKOFF = 3600.0
LEASE_MARGIN_S = 30.0
ATTEMPTS = 3
UPLOAD_TIMEOUT = aiohttp.ClientTimeout(total=120.0, sock_connect=15.0)


class UploadError(Exception):
    def __init__(self, status: int):
        super().__init__(
            f"upload returned HTTP {status}" if status else "upload failed"
        )
        self.status = status


class TaskWorker:
    """Leases one kind of task and runs up to max_tasks of them at once."""

    kind = ""

    def __init__(self, settings: Settings, api: TaskApiClient | None = None):
        self.settings = settings
        self.api = api or TaskApiClient(settings.task_api_url, settings.keypair())
        self.upload_http: aiohttp.ClientSession | None = None
        self.stopping = asyncio.Event()
        self.in_flight: set[asyncio.Task] = set()
        self.idle_polls = 0

    def stop(self) -> None:
        if self.stopping.is_set():
            for task in self.in_flight:
                task.cancel()
        self.stopping.set()

    async def run(self) -> None:
        stopped = asyncio.create_task(self.stopping.wait())
        try:
            while not stopped.done():
                if len(self.in_flight) >= self.settings.max_tasks:
                    await asyncio.wait(
                        {stopped, *self.in_flight}, return_when=asyncio.FIRST_COMPLETED
                    )
                    continue

                try:
                    delay = await self.poll()
                except Exception:
                    log.exception("lease poll failed")
                    delay = ERROR_BACKOFF
                if (
                    self.settings.idle_exit
                    and self.idle_polls >= self.settings.idle_exit
                ):
                    log.info("queue empty for %d polls, exiting", self.idle_polls)
                    break
                if delay:
                    await asyncio.wait({stopped}, timeout=delay)
            await self.drain()
        finally:
            stopped.cancel()
            await self.aclose()

    async def poll(self) -> float:
        try:
            answer = await self.api.post("/v1/tasks/lease", {"kind": self.kind})
        except TaskApiError as exc:
            log.warning("lease failed: %s", exc)
            return ERROR_BACKOFF
        if answer.get("receipt") and self.settings.receipts_file:
            await asyncio.to_thread(self.keep_receipt, answer["receipt"])

        task = answer.get("task")
        if task:
            self.idle_polls = 0
            job = asyncio.create_task(self.process_task(task))
            self.in_flight.add(job)
            job.add_done_callback(self.in_flight.discard)
            return 0.0

        refusal = answer.get("refusal") or {}
        code = refusal.get("code", "")
        if code == "QUEUE_EMPTY" and not self.in_flight:
            self.idle_polls += 1
        if code == "LOCKED_OUT":
            log.warning("locked out after failed verifications: %s", refusal["inputs"])
        try:
            return min(MAX_BACKOFF, max(0.05, float(refusal["inputs"]["retry_after"])))
        except (KeyError, TypeError, ValueError):
            return BACKOFF.get(code, ERROR_BACKOFF)

    async def process_task(self, task: dict) -> None:
        raise NotImplementedError

    async def upload(self, upload: dict, body: bytes) -> None:
        if self.upload_http is None:
            self.upload_http = aiohttp.ClientSession(timeout=UPLOAD_TIMEOUT)
        try:
            # Presigned URLs are sent byte for byte; re-quoting breaks the signature.
            async with self.upload_http.put(
                URL(upload["url"], encoded=True),
                data=body,
                headers={"Content-Type": upload["content_type"]},
            ) as response:
                status = response.status
        except (aiohttp.ClientError, asyncio.TimeoutError):
            raise UploadError(0) from None
        if status >= 400:
            raise UploadError(status)

    async def abandon(self, task_id: str, reason: str) -> None:
        log.warning("abandoning task %s: %s", task_id, reason)
        try:
            await self.api.post(f"/v1/tasks/{task_id}/abandon")
        except TaskApiError as exc:
            log.warning("could not abandon task %s: %s", task_id, exc)

    async def drain(self) -> None:
        if not self.in_flight:
            return
        grace = self.settings.shutdown_grace if self.stopping.is_set() else None
        log.info("waiting for %d in-flight tasks", len(self.in_flight))
        await asyncio.wait(self.in_flight, timeout=grace)
        for task in self.in_flight:
            task.cancel()
        await asyncio.gather(*self.in_flight, return_exceptions=True)

    def keep_receipt(self, receipt: dict) -> None:
        try:
            with open(self.settings.receipts_file, "a") as handle:
                handle.write(json.dumps(receipt, sort_keys=True) + "\n")
        except (OSError, ValueError) as exc:
            log.warning("could not keep receipt: %s", exc)

    async def aclose(self) -> None:
        if self.upload_http is not None:
            await self.upload_http.close()
        await self.api.aclose()


class Miner(TaskWorker):
    kind = "crawl"

    def __init__(
        self,
        settings: Settings,
        api: TaskApiClient | None = None,
        fetcher: Fetcher | None = None,
    ):
        super().__init__(settings, api)
        self.fetcher = fetcher or Fetcher(settings)
        self.scrapingdog = (
            ScrapingDog(
                settings.scrapingdog_api_key,
                settings.scrapingdog_concurrency,
                max_bytes=settings.max_bytes,
            )
            if settings.scrapingdog_api_key
            else None
        )
        # Extraction is CPU-bound and each parse holds a whole page tree in memory.
        self.extraction = ThreadPoolExecutor(
            settings.extraction_threads, thread_name_prefix="extract"
        )
        # A page holds its slot until it is in the upload, so pages never pile up.
        self.in_progress = asyncio.Semaphore(settings.concurrency)
        self.falling_back = asyncio.Semaphore(settings.scrapingdog_concurrency)

    async def process_task(self, task: dict) -> None:
        task_id, upload = task["task_id"], task["upload"]
        started = time.monotonic()
        pages = UploadWriter(task_id, self.api.hotkey)
        try:
            await self.crawl(task["urls"], task.get("expires_at"), pages)
            body = await pages.finish()
            await with_retries(lambda: self.upload(upload, body))
            report = {
                "key": upload["key"],
                "rows": pages.rows,
                "ok": pages.ok,
                "errors": pages.rows - pages.ok,
                "bytes": len(body),
            }
            await with_retries(
                lambda: self.api.post(f"/v1/tasks/{task_id}/complete", report)
            )
        except asyncio.CancelledError:
            await self.abandon(task_id, "shutting down")
            raise
        except Exception as exc:
            await self.abandon(task_id, f"{type(exc).__name__}: {exc}")
            return

        breakdown = " ".join(f"{name}={n}" for name, n in pages.errors.most_common())
        log.info(
            "task %s: %d urls, %d ok, %d errors%s, %.1fs, %d bytes uploaded",
            task_id,
            len(task["urls"]),
            pages.ok,
            pages.rows - pages.ok,
            f" ({breakdown})" if breakdown else "",
            time.monotonic() - started,
            len(body),
        )

    async def crawl(
        self, urls: list[str], expires_at: float | None, pages: UploadWriter
    ) -> None:
        deadline = None
        if expires_at:
            remaining = expires_at - time.time()
            deadline = expires_at - min(LEASE_MARGIN_S, remaining / 4)

        async def one(url: str) -> None:
            async with self.in_progress:
                row = await self.to_row(await self.fetcher.fetch(url, deadline))
                if not self.scrapingdog or not needs_fallback(
                    row["error"], row["status"]
                ):
                    await pages.add(row)
                    return
            # Off the crawl slot: ScrapingDog's latency would otherwise cap our own-IP rate.
            async with self.falling_back:
                await pages.add(await self.fallback_row(url, row, deadline))

        await asyncio.gather(*map(one, urls))

    async def fallback_row(self, url: str, row: dict, deadline: float | None) -> dict:
        fallback = await self.to_row(
            await self.scrapingdog.fetch(url, deadline=deadline)
        )
        return fallback if fallback["error"] is None else row

    async def to_row(self, fetched: Fetched) -> dict:
        try:
            return await asyncio.get_running_loop().run_in_executor(
                self.extraction, build_row, fetched, self.settings.max_bytes
            )
        except Exception as exc:
            log.warning(
                "row for %s failed: %s: %s", fetched.url, type(exc).__name__, exc
            )
            return error_row(fetched, "other")

    async def aclose(self) -> None:
        await self.fetcher.aclose()
        self.extraction.shutdown(wait=False, cancel_futures=True)
        if self.scrapingdog is not None:
            await self.scrapingdog.aclose()
        await super().aclose()


async def with_retries(call, attempts: int = ATTEMPTS):
    for attempt in range(1, attempts + 1):
        try:
            return await call()
        except (TaskApiError, UploadError) as exc:
            status = getattr(exc, "status", 0)
            if attempt == attempts or 0 < status < 500:
                raise
        await asyncio.sleep(attempt)


async def serve(miner: Miner | None = None) -> None:
    miner = miner or Miner(Settings.from_env())
    settings = miner.settings
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, miner.stop)

    log.info(
        "miner %s -> %s, %s, concurrency %d (%d per domain), up to %d tasks",
        miner.api.hotkey,
        settings.task_api_url,
        f"{len(settings.proxy_urls)} proxies" if settings.proxy_urls else "direct",
        settings.concurrency,
        settings.per_domain,
        settings.max_tasks,
    )
    await miner.run()


def main() -> None:
    env.load(ENV_FILE)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s"
    )
    logging.getLogger("trafilatura").setLevel(logging.ERROR)
    asyncio.run(serve())


if __name__ == "__main__":
    main()
