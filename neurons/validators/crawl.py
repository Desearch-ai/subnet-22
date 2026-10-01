from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable

import aiohttp

from desearch.kinds import CRAWL
from neurons.validators.scoring import (
    MATCH_RATIO,
    MIN_SAMPLES,
    NOT_FETCHED,
    FetchedPage,
    empty_result,
)
from neurons.validators.scoring_process import (
    MEMORY_MB,
    ScoringProcess,
    Unscorable,
)
from neurons.validators.tasks import (
    MAX_DOWNLOAD_BYTES,
    DownloadFailed,
    TaskChecker,
    UploadMissing,
)

PageFetcher = Callable[..., Awaitable[FetchedPage]]

SCORE_TIMEOUT_S = 300.0
PAGE_TIMEOUT_S = 60.0
CHECK_ATTEMPTS = 2

log = logging.getLogger("validator")


class CrawlValidator(TaskChecker):
    kinds = (CRAWL,)
    warned_unconfined = False

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
        ledger=None,
        storage_url: str = "",
        seeds=None,
        signer: str = "",
    ):
        super().__init__(api, http, max_download, ledger, storage_url, seeds, signer)
        self.fetcher = fetcher
        self.min_samples = min_samples
        self.match_ratio = match_ratio
        self.score_timeout = score_timeout
        self.memory_mb = memory_mb

    async def check(self, job: dict) -> dict | None:
        try:
            result = await self.score_task(job)
        except UploadMissing:
            log.warning("task=%s upload is gone", job["task_id"])
            await self.hand_back(job["task_id"], "missing")
            return {}
        except DownloadFailed as exc:
            log.warning("%s, trying again later", exc)
            self.defer(job["task_id"])
            return {}

        if result["verdict"] == "retry":
            log.warning(
                "task=%s ScrapingDog failed %d of %d samples, trying again later",
                job["task_id"],
                result[NOT_FETCHED],
                result["sampled"],
            )
            self.provider_failed()
            self.defer(job["task_id"])
            return {}
        self.provider_worked()
        if result["reason"] == "unscorable":
            self.scoring_failed()
        else:
            self.scoring_worked()

        await self.submit_verdict(job, result)
        self.note_verdict(job, result)
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
        data = await self.download(job["download_url"], job["task_id"]) or b""
        for attempt in range(CHECK_ATTEMPTS):
            try:
                result = await self.run_check(job, data)
                break
            except Unscorable as why:
                log.warning("task=%s could not be scored: %s", job["task_id"], why)
                crashed = str(why) == "died"
                if crashed and attempt + 1 < CHECK_ATTEMPTS:
                    continue
                result = empty_result(set(job["urls"]), "unscorable")
                # A file that crashes a checker which scores other uploads fine is the miner's doing.
                if crashed and self.healthy():
                    result["crashed"] = True
                break
        return {**result, "took_ms": int((time.monotonic() - started) * 1000)}

    async def run_check(self, job: dict, data: bytes) -> dict:
        """Fetches every sample page at once, then a rendered copy of those that need one."""
        session = ScoringProcess(
            data,
            job["urls"],
            job["seed"],
            self.min_samples,
            self.match_ratio,
            self.score_timeout,
            self.memory_mb,
        )
        try:
            await session.start()
            planned = await session.wait_for("samples")
            self.note_confinement(planned["unconfined"])
            urls = planned["urls"]
            pages = await asyncio.gather(*(self._fetch_sample(url) for url in urls))
            doubtful = await session.ask("doubtful", dict(zip(urls, pages, strict=True)))
            rendered = await asyncio.gather(
                *(self._fetch_sample(url, rendered=True) for url in doubtful)
            )
            return await session.ask("scored", dict(zip(doubtful, rendered, strict=True)))
        finally:
            await session.aclose()

    def note_confinement(self, unconfined: list[str]) -> None:
        if unconfined and not CrawlValidator.warned_unconfined:
            CrawlValidator.warned_unconfined = True
            log.warning(
                "The check runs without %s isolation on this system; see the validator"
                " setup guide",
                " or ".join(unconfined),
            )

    async def _fetch_sample(self, url: str, rendered: bool = False) -> FetchedPage:
        """A page the validator cannot load in time counts as unverifiable, never against the miner."""
        try:
            fetch = self.fetcher(url, rendered=True) if rendered else self.fetcher(url)
            return await asyncio.wait_for(fetch, PAGE_TIMEOUT_S)
        except asyncio.TimeoutError:
            return FetchedPage(0, error="timeout")
        except Exception as exc:
            return FetchedPage(0, error=f"fetcher_{type(exc).__name__}")
