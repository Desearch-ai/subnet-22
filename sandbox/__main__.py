"""python -m sandbox: a local task API and validator to test your miner against."""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import time
from pathlib import Path

from desearch import env
from desearch.kinds import CRAWL, EMBED
from sandbox import local
from sandbox.local import ROOT, Ports

QUICK_TASK_SIZE = 100


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    parser = argparse.ArgumentParser(prog="python -m sandbox")
    parser.add_argument(
        "mode",
        nargs="?",
        choices=(CRAWL, EMBED),
        default=CRAWL,
        help="crawl tasks, the default, or embed tasks",
    )
    parser.add_argument(
        "--urls",
        type=Path,
        default=None,
        help="your own URLs to serve: a parquet file with a url column, or one URL per line",
    )
    parser.add_argument(
        "--embed-set",
        type=Path,
        default=None,
        help="a folder holding an embed test set, instead of the public one",
    )
    parser.add_argument(
        "--task-size",
        type=int,
        default=QUICK_TASK_SIZE,
        help="URLs per crawl task or pages per embed task: 100 for quick feedback, 1000 to match mainnet",
    )
    parser.add_argument("--port", type=int, default=18080, help="the task API port")
    parser.add_argument("--runs", type=Path, default=ROOT / "sandbox" / "runs")
    args = parser.parse_args(argv)

    if args.mode == EMBED:
        return asyncio.run(run_embed(args))

    from sandbox.run import Sandbox
    from sandbox.urls import DatasetUrls, FileUrls

    if not load_scrapingdog_key():
        print(
            "SCRAPINGDOG_API_KEY is required, as for a mainnet validator: set it in the"
            " shell or in neurons/validators/.env or neurons/miners/.env",
            flush=True,
        )
        return 1
    run_dir = new_run_dir(args.runs)
    source = (
        FileUrls(args.urls) if args.urls else DatasetUrls(ROOT / "sandbox" / "cache")
    )
    sandbox = Sandbox(Ports(args.port), run_dir, source, args.task_size)
    return asyncio.run(sandbox.run())


async def run_embed(args: argparse.Namespace) -> int:
    from sandbox.embed import EmbedSandbox, EmbedSet, download_set

    folder = args.embed_set or await download_set()
    texts = await asyncio.to_thread(EmbedSet.read, folder)
    sandbox = EmbedSandbox(
        Ports(args.port), new_run_dir(args.runs), texts, args.task_size
    )
    return await sandbox.run()


def new_run_dir(runs: Path) -> Path:
    run_dir = runs / time.strftime("%Y%m%d-%H%M%S")
    run_dir.mkdir(parents=True)
    return run_dir


def load_scrapingdog_key() -> bool:
    """The local validator's ScrapingDog key, from the shell or either neuron's .env file."""
    if os.environ.get("SCRAPINGDOG_API_KEY"):
        return True
    for path in local.ENV_FILES:
        found = env.read_dotenv(path).get("SCRAPINGDOG_API_KEY")
        if found:
            os.environ["SCRAPINGDOG_API_KEY"] = found
            return True
    return False


if __name__ == "__main__":
    sys.exit(main())
