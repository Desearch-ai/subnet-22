from __future__ import annotations

import asyncio
import hashlib
import logging
import signal
import time
from collections import deque

import aiohttp
from yarl import URL

from desearch import env
from desearch.embedding import (
    INPUT_SCHEMA,
    OUTPUT_SCHEMA,
    EmbeddingClient,
    encode_vector,
    model_named,
    read_parquet,
    write_parquet,
)
from desearch.kinds import EMBED
from neurons.miners.config import ENV_FILE, Settings
from neurons.miners.miner import HEADROOM, TaskWorker, with_retries

log = logging.getLogger("embed_miner")

DOWNLOAD_TIMEOUT = aiohttp.ClientTimeout(total=120.0, sock_connect=15.0)
DOWNLOAD_ATTEMPTS = 3
# Kept before the claim expires for the upload and the completion call.
UPLOAD_MARGIN_S = 20.0
SERVER_WAIT_S = 10.0
PACE_TASKS = 5


class EmbedMiner(TaskWorker):
    kind = EMBED

    def __init__(self, settings: Settings, api=None, embedder=None):
        super().__init__(settings, api)
        self.embedder = embedder or EmbeddingClient(
            settings.embed_api_url,
            model_named(settings.embed_model),
            settings.embed_api_key,
        )
        self.download_http: aiohttp.ClientSession | None = None
        # One task at a time on the model server: sharing it only makes every task late.
        self.server = asyncio.Lock()
        self.ahead: dict[str, int] = {}
        self.recent: deque[tuple[int, float]] = deque(maxlen=PACE_TASKS)
        self.window = 0.0
        self.task_texts = 0

    def claim_body(self) -> dict:
        return {**super().claim_body(), "models": [self.embedder.model.name]}

    def claimed(self, task: dict) -> None:
        self.ahead[task["task_id"]] = task.get("texts", 0)
        self.task_texts = max(self.task_texts, task.get("texts", 0))
        self.window = task["expires_at"] - time.time() - UPLOAD_MARGIN_S

    def has_room(self) -> bool:
        """Another task is claimed only if, at the measured pace, every held one still finishes in time."""
        ahead = sum(self.ahead.values())
        if not ahead:
            return True
        pace = self.pace()
        if not pace:
            return False
        return (ahead + self.task_texts) / pace <= HEADROOM * self.window

    def pace(self) -> float | None:
        """Texts a second the model server embedded over the last few tasks."""
        seconds = sum(took for _, took in self.recent)
        return sum(texts for texts, _ in self.recent) / seconds if seconds else None

    async def process_task(self, task: dict) -> None:
        task_id, upload = task["task_id"], task["upload"]
        started = time.monotonic()
        try:
            served = self.embedder.model.name
            if task["model"] != served:
                raise ValueError(f"asked for {task['model']}, this miner runs {served}")
            left = task["expires_at"] - time.time() - UPLOAD_MARGIN_S
            rows, out = await asyncio.wait_for(self.embedded(task), max(left, 0.0))
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
        except asyncio.TimeoutError:
            await self.abandon(task_id, "could not embed it before the claim expires")
            return
        except Exception as exc:
            await self.abandon(task_id, f"{type(exc).__name__}: {exc}")
            return
        finally:
            self.ahead.pop(task_id, None)
        log.info(
            "task %s: %d texts embedded in %.1fs",
            task_id,
            len(rows),
            time.monotonic() - started,
        )

    async def embedded(self, task: dict) -> tuple[list[dict], bytes]:
        """The task's input rows and the upload of their vectors."""
        body = await self.download(task["input"]["url"])
        if hashlib.sha256(body).hexdigest() != task["input"]["sha256"]:
            raise ValueError("the input does not match the hash it was issued with")
        rows = read_parquet(body, INPUT_SCHEMA)
        async with self.server:
            started = time.monotonic()
            vectors = await self.embedder.embed([row["text"] for row in rows])
            self.recent.append((len(rows), time.monotonic() - started))
        self.ahead.pop(task["task_id"], None)
        out = write_parquet(
            [
                {"text_id": row["text_id"], "vector": encode_vector(vector)}
                for row, vector in zip(rows, vectors, strict=True)
            ],
            OUTPUT_SCHEMA,
        )
        return rows, out

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

    async def server_ready(self) -> bool:
        """Waits for the model server; False when it serves a different model."""
        name, url = self.embedder.model.name, self.settings.embed_api_url
        while not self.stopping.is_set():
            try:
                served = await self.embedder.served_models()
            except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
                log.info("waiting for the model server at %s: %r", url, exc)
                await asyncio.sleep(SERVER_WAIT_S)
                continue
            if name in served:
                return True
            log.error(
                "the model server at %s serves %s, not %s; start it with --served-model-name %s",
                url,
                served,
                name,
                name,
            )
            return False
        return False

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
    if not await miner.server_ready():
        await miner.aclose()
        return
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
