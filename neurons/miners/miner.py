from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import signal
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import aiohttp
from yarl import URL

from desearch import env
from desearch.client import TaskApiClient, TaskApiError
from desearch.fetch import (
    Fetched,
    Fetcher,
    ScrapingDog,
    needs_fallback,
    worth_another_try,
)
from desearch.kinds import CRAWL
from neurons.miners.config import ENV_FILE, Settings
from neurons.miners.rows import UploadWriter, build_row, error_row

log = logging.getLogger("miner")

BACKOFF = {"QUEUE_EMPTY": 2.0, "NO_CAPACITY": 1.0}
ERROR_BACKOFF = 2.0
MAX_BACKOFF = 3600.0
CLAIM_MARGIN_S = 30.0
UPLOAD_SLACK = 2.0
PACE_WINDOW_S = 30.0
PACE_MIN_S = 15.0
HEADROOM = 0.7
ROOM_CHECK_S = 1.0
ATTEMPTS = 3
SUMMARY_EVERY_S = 60.0
UPLOAD_TIMEOUT = aiohttp.ClientTimeout(total=120.0, sock_connect=15.0)


class UploadError(Exception):
    def __init__(self, status: int):
        super().__init__(
            f"upload returned HTTP {status}" if status else "upload failed"
        )
        self.status = status


class TaskWorker:
    """Claims one kind of task and runs up to max_tasks of them at once."""

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
                if not self.has_room():
                    await asyncio.wait(
                        {stopped, *self.in_flight},
                        timeout=ROOM_CHECK_S,
                        return_when=asyncio.FIRST_COMPLETED,
                    )
                    continue

                try:
                    delay = await self.poll()
                except Exception:
                    log.exception("claim poll failed")
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
        body = {"kind": self.kind, "count": self.wanted()}
        try:
            answer = await self.api.post("/v1/tasks/claim", body)
        except TaskApiError as exc:
            log.warning("claim failed: %s", exc)
            return ERROR_BACKOFF
        receipts = answer.get("receipts") or [answer.get("receipt")]
        if self.settings.receipts_file:
            for receipt in filter(None, receipts):
                await asyncio.to_thread(self.keep_receipt, receipt)

        tasks = answer.get("tasks") or []
        for task in tasks:
            self.claimed(task)
            job = asyncio.create_task(self.process_task(task))
            self.in_flight.add(job)
            job.add_done_callback(self.in_flight.discard)
        if tasks:
            self.idle_polls = 0
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

    def has_room(self) -> bool:
        return True

    def wanted(self) -> int:
        return 1

    def claimed(self, task: dict) -> None:
        pass

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


class Throughput:
    """Tasks and pages uploaded, since start and since the last summary."""

    def __init__(self, now: float):
        self.started = self.since = now
        self.tasks = self.pages = self.ok = 0
        self.recent_tasks = self.recent_pages = self.recent_ok = 0

    def add(self, pages: int, ok: int) -> None:
        self.tasks += 1
        self.pages += pages
        self.ok += ok
        self.recent_tasks += 1
        self.recent_pages += pages
        self.recent_ok += ok

    def line(self, now: float) -> str:
        window = max(now - self.since, 1e-9)
        text = (
            f"last {window:.0f}s: {self.recent_tasks} tasks, {self.recent_pages} pages"
            f" ({self.recent_ok} ok), {self.recent_pages / window:.1f} pages/s;"
            f" since start: {self.tasks} tasks, {self.pages} pages ({self.ok} ok),"
            f" {self.pages / max(now - self.started, 1e-9):.1f} pages/s"
        )
        self.since = now
        self.recent_tasks = self.recent_pages = self.recent_ok = 0
        return text


class Pace:
    """Pages a second all crawl slots can take, from how long pages held one in the last half minute."""

    def __init__(self, slots: int):
        self.slots = slots
        self.finished: deque[tuple[float, float]] = deque()

    def add(self, now: float, held_s: float) -> None:
        self.finished.append((now, held_s))

    def per_second(self, now: float) -> float | None:
        while self.finished and self.finished[0][0] < now - PACE_WINDOW_S:
            self.finished.popleft()
        if not self.finished or now - self.finished[0][0] < PACE_MIN_S:
            return None
        held = sum(seconds for _, seconds in self.finished)
        return self.slots * len(self.finished) / max(held, 1e-3)


@dataclass
class Progress:
    left: int
    late: int = 0


class Miner(TaskWorker):
    kind = CRAWL

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
        self.awaiting_fallback = 0
        self.throughput = Throughput(time.monotonic())
        self.pace = Pace(settings.concurrency)
        self.crawls: dict[str, Progress] = {}
        self.task_urls = 0
        self.crawl_window = math.inf
        self.upload_s = 0.0

    def claimed(self, task: dict) -> None:
        self.crawls[task["task_id"]] = Progress(len(task["urls"]))
        self.task_urls = len(task["urls"])
        if task.get("expires_at"):
            deadline = self.crawl_deadline(task["expires_at"])
            self.crawl_window = deadline - time.time()

    def has_room(self) -> bool:
        """Another task is claimed only if, at the current pace, all of them still finish in time."""
        ahead = sum(crawl.left for crawl in self.crawls.values())
        if not ahead:
            return True
        # The pace counts our own slots only; a queue for ScrapingDog means it is the limit.
        if self.awaiting_fallback > self.settings.scrapingdog_concurrency:
            return False
        pace = self.pace.per_second(time.monotonic())
        if not pace:
            return False
        return (ahead + self.task_urls) / pace <= HEADROOM * self.crawl_window

    def wanted(self) -> int:
        """As many tasks as the measured pace says will still finish in time."""
        free = max(1, self.settings.max_tasks - len(self.in_flight))
        pace = self.pace.per_second(time.monotonic())
        if not pace or not self.task_urls:
            return 1
        if math.isinf(self.crawl_window):
            return free
        ahead = sum(crawl.left for crawl in self.crawls.values())
        fits = int((HEADROOM * self.crawl_window * pace - ahead) // self.task_urls)
        return max(1, min(free, fits))

    def crawl_deadline(self, expires_at: float) -> float:
        """The end of the claim is kept for writing and uploading the file."""
        margin = max(CLAIM_MARGIN_S, UPLOAD_SLACK * self.upload_s)
        return expires_at - min(margin, (expires_at - time.time()) / 2)

    async def run(self) -> None:
        summaries = asyncio.create_task(self.summarize())
        try:
            await super().run()
        finally:
            summaries.cancel()

    async def summarize(self) -> None:
        while True:
            await asyncio.sleep(SUMMARY_EVERY_S)
            log.info(self.throughput.line(time.monotonic()))

    async def process_task(self, task: dict) -> None:
        task_id, upload = task["task_id"], task["upload"]
        started = time.monotonic()
        pages = UploadWriter(task_id, self.api.hotkey)
        progress = self.crawls.get(task_id)
        try:
            await self.crawl(task["urls"], task.get("expires_at"), pages, progress)
            crawled = time.monotonic()
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
            self.upload_s = max(time.monotonic() - crawled, 0.8 * self.upload_s)
        except asyncio.CancelledError:
            await self.abandon(task_id, "shutting down")
            raise
        except Exception as exc:
            await self.abandon(task_id, f"{type(exc).__name__}: {exc}")
            return
        finally:
            self.crawls.pop(task_id, None)

        self.throughput.add(pages.rows, pages.ok)
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
        self,
        urls: list[str],
        expires_at: float | None,
        pages: UploadWriter,
        progress: Progress | None = None,
    ) -> None:
        deadline = self.crawl_deadline(expires_at) if expires_at else None
        progress = progress or Progress(len(urls))

        async def one(url: str) -> None:
            try:
                row = await self.own_row(url, deadline, pages, progress)
                if row is None:
                    return
                # Off the crawl slot: ScrapingDog's latency would otherwise cap our own-IP rate.
                row = await self.with_fallback(url, row, deadline)
                log_page(row)
                await pages.add(row)
            finally:
                progress.left -= 1

        await asyncio.gather(*map(one, urls))
        if progress.late:
            log.warning(
                "%d of %d URLs were not fetched before the deadline;"
                " raise CRAWL_CONCURRENCY or lower MAX_TASKS",
                progress.late,
                len(urls),
            )

    async def own_row(
        self,
        url: str,
        deadline: float | None,
        pages: UploadWriter,
        progress: Progress,
    ) -> dict | None:
        """Fetches from our own addresses and writes the row, or returns it for the fallback."""
        settings = self.settings
        timeouts = [settings.first_timeout] + [settings.timeout] * (
            settings.attempts - 1
        )
        held = 0.0
        for tries_left, timeout in zip(reversed(range(len(timeouts))), timeouts):
            # A retry takes a new slot, so it waits behind the pages not tried yet.
            async with self.in_progress:
                late = out_of_time(deadline)
                started = time.monotonic()
                row = await self.to_row(
                    await self.fetcher.attempt(url, deadline, timeout)
                )
                held += time.monotonic() - started
                again = (
                    tries_left
                    and not out_of_time(deadline)
                    and worth_another_try(
                        row["error"], row["status"], self.fetcher.rerouting
                    )
                )
                if again:
                    continue
                progress.late += late
                # A page cut off by the deadline ends at once and would overstate the pace.
                if not late:
                    self.pace.add(time.monotonic(), held)
                if self.scrapingdog and needs_fallback(row["error"], row["status"]):
                    return row
                log_page(row)
                await pages.add(row)
                return None

    async def with_fallback(self, url: str, row: dict, deadline: float | None) -> dict:
        """ScrapingDog's page if a fallback slot frees before the deadline, else our own row."""
        wait = None if deadline is None else max(0.0, deadline - time.time())
        acquiring = asyncio.ensure_future(self.falling_back.acquire())
        self.awaiting_fallback += 1
        try:
            await asyncio.wait({acquiring}, timeout=wait)
        except BaseException:
            if acquiring.done() and not acquiring.cancelled():
                self.falling_back.release()
            acquiring.cancel()
            raise
        finally:
            self.awaiting_fallback -= 1
        if not acquiring.done():
            acquiring.cancel()
            return row
        try:
            return await self.fallback_row(url, row, deadline)
        finally:
            self.falling_back.release()

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


def out_of_time(deadline: float | None) -> bool:
    return deadline is not None and time.time() >= deadline


def log_page(row: dict) -> None:
    if log.isEnabledFor(logging.DEBUG):
        outcome = row["error"] or f"{len(row.get('text') or '')} chars of text"
        log.debug("page %s %s: %s", row.get("status"), row["url"], outcome)


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
        level=os.environ.get("LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    logging.getLogger("trafilatura").setLevel(logging.ERROR)
    asyncio.run(serve())


if __name__ == "__main__":
    main()
