"""python -m bulk: Wikipedia snapshots into a folder of batch files for `publisher bulk`."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from bulk import wikipedia
from bulk.rows import Batches, parse_as_of


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m bulk")
    commands = parser.add_subparsers(dest="command", required=True)
    found = commands.add_parser(
        "wikipedia",
        help="Wikimedia Enterprise snapshots of English Wikipedia and its sister projects",
    )
    found.add_argument(
        "--out",
        type=Path,
        required=True,
        help="a new folder for this run's batch files",
    )
    found.add_argument(
        "--as-of",
        required=True,
        help="when the source's data was taken, as an ISO time with a zone; the pages' fetch time",
    )
    found.add_argument(
        "--run",
        default="",
        help="the run's name in its task id, bulk-<source>-<run>; the as-of date by default",
    )
    found.add_argument(
        "--limit", type=int, default=0, help="stop after this many pages, for trials"
    )
    found.add_argument("files", type=Path, nargs="+")
    found.add_argument(
        "--hidden",
        type=Path,
        required=True,
        help="hidden category names, one per line, from `hidden-categories`",
    )
    found.add_argument(
        "--workers",
        type=int,
        default=os.cpu_count() or 1,
        help="processes that frame and extract articles",
    )
    hidden = commands.add_parser(
        "hidden-categories",
        help="print the hidden categories of a category-namespace snapshot",
    )
    hidden.add_argument("files", type=Path, nargs="+")

    args = parser.parse_args(sys.argv[1:] if argv is None else argv)
    if args.command == "hidden-categories":
        for path in args.files:
            for name in wikipedia.hidden_categories(wikipedia.articles(path)):
                print(name)
        return 0

    as_of = parse_as_of(args.as_of)
    batches = Batches(
        args.out, args.command, args.run or as_of.strftime("%Y%m%d"), as_of
    )
    hidden_names = set(args.hidden.read_text().splitlines())
    counts = wikipedia.run(
        args.files, batches, as_of, hidden_names, args.limit, args.workers
    )
    batches.close(dict(counts))
    print(
        json.dumps(
            {
                "out": str(args.out),
                "files": len(batches.files),
                "rows": batches.rows,
                "removed": len(batches.removed),
                **counts,
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
