from __future__ import annotations

import asyncio
import hashlib
import logging
import signal
import time

import aiohttp
from yarl import URL

from desearch import env
from desearch.embedding import (
    INPUT_SCHEMA,
    OUTPUT_SCHEMA,
    HostedEmbedder,
    encode_vector,
    model_named,
    read_parquet,
    write_parquet,
)
from neurons.miners.config import ENV_FILE, Settings
from neurons.miners.miner import TaskWorker, with_retries

log = logging.getLogger("embed_miner")

DOWNLOAD_TIMEOUT = aiohttp.ClientTimeout(total=120.0, sock_connect=15.0)
DOWNLOAD_ATTEMPTS = 3


class EmbedMiner(TaskWorker):
    kind = "embed"

    def __init__(self, settings: Settings, api=None, embedder=None):
        super().__init__(settings, api)
        self.embedder = embedder or HostedEmbedder(
            settings.embed_api_url,
            settings.embed_api_key,
            model_named(settings.embed_model),
            settings.embed_providers,
        )
        self.download_http: aiohttp.ClientSession | None = None

    async def process_task(self, task: dict) -> None:
        task_id, upload = task["task_id"], task["upload"]
        started = time.monotonic()
        try:
            served = self.embedder.model.name
            if task["model"] != served:
                raise ValueError(f"asked for {task['model']}, this miner runs {served}")
            body = await self.download(task["input"]["url"])
            if hashlib.sha256(body).hexdigest() != task["input"]["sha256"]:
                raise ValueError("the input does not match the hash it was issued with")
            rows = read_parquet(body, INPUT_SCHEMA)
            vectors = await self.embedder.embed([row["text"] for row in rows])
            out = write_parquet(
                [
                    {"text_id": row["text_id"], "vector": encode_vector(vector)}
                    for row, vector in zip(rows, vectors, strict=True)
                ],
                OUTPUT_SCHEMA,
            )
            await with_retries(lambda: self.upload(upload, out))
            report = {
                "key": upload["key"],
                "rows": len(rows),
                "ok": len(rows),
                "errors": 0,
                "bytes": len(out),
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
        log.info(
            "task %s: %d texts embedded in %.1fs",
            task_id,
            len(rows),
            time.monotonic() - started,
        )

    async def download(self, url: str) -> bytes:
        if self.download_http is None:
            self.download_http = aiohttp.ClientSession(timeout=DOWNLOAD_TIMEOUT)
        for attempt in range(1, DOWNLOAD_ATTEMPTS + 1):
            try:
                async with self.download_http.get(URL(url, encoded=True)) as response:
                    response.raise_for_status()
                    return await response.read()
            except (aiohttp.ClientError, asyncio.TimeoutError):
                if attempt == DOWNLOAD_ATTEMPTS:
                    raise
                await asyncio.sleep(attempt)
        raise AssertionError("unreachable")

    async def aclose(self) -> None:
        await self.embedder.aclose()
        if self.download_http is not None:
            await self.download_http.close()
        await super().aclose()


async def serve(miner: EmbedMiner | None = None) -> None:
    miner = miner or EmbedMiner(Settings.from_env())
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, miner.stop)
    log.info(
        "embed miner %s -> %s, %s, up to %d tasks",
        miner.api.hotkey,
        miner.settings.task_api_url,
        miner.embedder.model.name,
        miner.settings.max_tasks,
    )
    await miner.run()


def main() -> None:
    env.load(ENV_FILE)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s"
    )
    asyncio.run(serve())


if __name__ == "__main__":
    main()
