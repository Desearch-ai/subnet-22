"""python -m sandbox: a local task API and validator to test your miner against."""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import time
from pathlib import Path

from desearch import env
from sandbox import local
from sandbox.local import ROOT, Ports

QUICK_TASK_SIZE = 100


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    parser = argparse.ArgumentParser(prog="python -m sandbox")
    parser.add_argument(
        "--urls",
        type=Path,
        default=None,
        help="your own URLs to serve: a parquet file with a url column, or one URL per line",
    )
    parser.add_argument(
        "--task-size",
        type=int,
        default=QUICK_TASK_SIZE,
        help="URLs per task: 100 for quick feedback, 1000 to match mainnet",
    )
    parser.add_argument("--port", type=int, default=18080, help="the task API port")
    parser.add_argument("--runs", type=Path, default=ROOT / "sandbox" / "runs")
    nodes = parser.add_subparsers(dest="node")
    validator = nodes.add_parser("validator", help=argparse.SUPPRESS)
    validator.add_argument("--port", type=int, required=True)
    validator.add_argument("--run-dir", type=Path, required=True)
    validator.add_argument("--genesis", type=float, required=True)
    args = parser.parse_args(argv)

    if args.node == "validator":
        from sandbox.validator import main as validator_main

        return validator_main(Ports(args.port), args.run_dir, args.genesis)

    from sandbox.run import Sandbox
    from sandbox.urls import DatasetUrls, FileUrls

    if not load_scrapingdog_key():
        print(
            "SCRAPINGDOG_API_KEY is required, as for a mainnet validator: set it in the"
            " shell or in neurons/validators/.env or neurons/miners/.env",
            flush=True,
        )
        return 1
    run_dir = args.runs / time.strftime("%Y%m%d-%H%M%S")
    run_dir.mkdir(parents=True)
    source = (
        FileUrls(args.urls) if args.urls else DatasetUrls(ROOT / "sandbox" / "cache")
    )
    sandbox = Sandbox(Ports(args.port), run_dir, source, args.task_size)
    return asyncio.run(sandbox.run())


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
