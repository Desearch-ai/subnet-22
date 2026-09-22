from __future__ import annotations

import asyncio
import logging
import secrets
import time
from collections.abc import Awaitable, Callable

import aiohttp

from neurons.validators.scoring import (
    MATCH_RATIO,
    MIN_SAMPLES,
    NOT_FETCHED,
    FetchedPage,
    empty_result,
    sample_seed,
)
from neurons.validators.scoring_process import MEMORY_MB, ScoringProcess, Unscorable
from neurons.validators.tasks import (
    MAX_DOWNLOAD_BYTES,
    DownloadFailed,
    TaskChecker,
    UploadMissing,
)

PageFetcher = Callable[..., Awaitable[FetchedPage]]

SCORE_TIMEOUT_S = 300.0

log = logging.getLogger("validator")

# Keeps the sample unpredictable to miners.
SAMPLE_SALT = secrets.token_hex(32)


class CrawlValidator(TaskChecker):
    kinds = ("crawl",)

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
        super().__init__(api, http, max_download)
        self.fetcher = fetcher
        self.min_samples = min_samples
        self.match_ratio = match_ratio
        self.score_timeout = score_timeout
        self.memory_mb = memory_mb

    async def check_next_task(self) -> dict | None:
        job = await self.lease()
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
            log.warning(
                "task=%s ScrapingDog failed %d of %d samples, handing it back",
                job["task_id"],
                result[NOT_FETCHED],
                result["sampled"],
            )
            self.provider_failed()
            await self.hand_back(job["task_id"], "provider")
            return {}
        self.provider_worked()

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
        data = await self.download(job["download_url"], job["task_id"])
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

    async def _fetch_sample(self, url: str, rendered: bool = False) -> FetchedPage:
        try:
            if rendered:
                return await self.fetcher(url, rendered=True)
            return await self.fetcher(url)
        except Exception as exc:
            return FetchedPage(0, error=f"fetcher_{type(exc).__name__}")
