"""Feeds the queue from desearch-bot's stores, opened as a RocksDB secondary."""

from __future__ import annotations

import argparse
import asyncio
import sys

from feeder import loop
from feeder.bot_stores import BUCKETS_DIR, SECONDARY_DIR, collect_pages


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m feeder", description=__doc__)
    parser.add_argument("--api", default="http://localhost:8080")
    parser.add_argument("--key-uri", default="//admin")
    parser.add_argument("--buckets", default=BUCKETS_DIR)
    parser.add_argument("--secondary", default=SECONDARY_DIR)
    parser.add_argument("--domains", default="feeder/news_domains.json")
    parser.add_argument("--domains-limit", type=int, default=10000)
    parser.add_argument("--per-domain", type=int, default=25)
    parser.add_argument("--interval", type=float, default=3600)
    parser.add_argument("--batch-target", type=int, default=25)
    parser.add_argument("--queue-cap", type=int, default=loop.QUEUE_CAP)
    parser.add_argument("--low-water", type=int, default=loop.LOW_WATER)
    parser.add_argument("--refresh", type=float, default=loop.REFRESH_S)
    parser.add_argument("--state", default="feeder/sent.db")
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args(argv)

    hosts = loop.load_domains(args.domains, args.domains_limit)

    async def source() -> list[dict]:
        return await asyncio.to_thread(
            collect_pages, args.buckets, args.secondary, hosts, args.per_domain
        )

    return asyncio.run(loop.run(args, source))


if __name__ == "__main__":
    sys.exit(main())
