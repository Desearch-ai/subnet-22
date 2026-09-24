from __future__ import annotations

import asyncio
import logging
import os
import signal

import redis.asyncio as aioredis
from app.queues import PublishQueue
from app.storage import Storage

from publisher.worker import Publisher

log = logging.getLogger("publisher")


def same_bucket(temp, pages) -> None:
    """The temp bucket expires daily; publishing there would lose records."""
    if (temp.bucket, temp.prefix) == (pages.bucket, pages.prefix):
        raise SystemExit(
            f"CF_R2_PAGES_BUCKET is {pages.bucket!r}, the same place uploads are kept and expired;"
            " point it at the permanent bucket"
        )


async def serve() -> None:
    redis = aioredis.from_url(
        os.environ.get("TASK_API_REDIS", "redis://localhost:6379/15"),
        decode_responses=True,
    )
    temp = Storage()
    pages = Storage(
        bucket=os.environ.get("CF_R2_PAGES_BUCKET", "desearch-pages"),
        prefix=os.environ.get("CF_R2_PAGES_PREFIX", ""),
    )
    same_bucket(temp, pages)
    for storage in (temp, pages):
        await storage.check()

    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, stop.set)

    publisher = Publisher(
        PublishQueue(
            redis,
            int(os.environ.get("TASK_API_PUBLISH_TTL", "600")),
            int(os.environ.get("TASK_API_PUBLISH_TRIES", "5")),
        ),
        temp,
        pages,
        workers=int(os.environ.get("PUBLISHER_WORKERS", "32")),
        batch=int(os.environ.get("PUBLISHER_BATCH", "20")),
    )
    log.info("publishing %s -> %s", temp.bucket, pages.bucket)
    try:
        await publisher.run(stop, int(os.environ.get("PUBLISHER_IDLE_EXIT", "0")))
    finally:
        publisher.close()
        await redis.aclose()
    log.info("publisher stopped")


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s"
    )
    asyncio.run(serve())


if __name__ == "__main__":
    main()
