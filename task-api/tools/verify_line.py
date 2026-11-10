#!/usr/bin/env python3
"""Checks across rounds that one kind of task went to the places that had waited longest; needs no credentials."""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "verify_round", Path(__file__).with_name("verify_round.py")
)
verify_round = importlib.util.module_from_spec(spec)
spec.loader.exec_module(verify_round)


def issues(api: str, kind: str, count: int) -> list[dict]:
    """The newest issues of one kind from the public events, oldest first: each kind has a line of its own."""
    found, before = [], None
    while len(found) < count:
        path = f"/v1/events?outcome=issued&kind={kind}&limit=100"
        page = verify_round.get(api, path + (f"&before={before}" if before else ""))
        found += page["events"]
        before = page["next"]
        if not before:
            break
    return list(reversed(found[:count]))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--api", default="http://localhost:8080")
    parser.add_argument("--kind", choices=["embed", "crawl"], default="embed")
    parser.add_argument("--issues", type=int, default=2000)
    parser.add_argument("--keep", type=float, default=verify_round.LINE_KEEP_S)
    args = parser.parse_args()
    entries = [
        {**event, "seq": event["id"], "served_at": event["at"]}
        for event in issues(args.api, args.kind, args.issues)
    ]
    ok, why = verify_round._line_order(entries, args.keep)
    print(("PASS: " if ok else "FAIL: ") + why)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
