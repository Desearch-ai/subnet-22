#!/usr/bin/env python3
"""What the planner did in each few minutes, from its report lines on stdin, to see what stops when embed supply falls.

    docker logs --timestamps --since 4h PLANNER 2>&1 | grep 'planner: through' | python3 planner_swing.py
"""

from __future__ import annotations

import argparse
import re
import sys
from datetime import datetime

WATCHED = (
    "passage_texts",
    "backfill",
    "shards",
    "shard_failures",
    "shards_skipped",
    "passage_files",
    "bulk_files",
    "over_budget",
    "no_room",
    "log_held",
    "handed_over",
    # Only in reports from builds that have them.
    "backfill_starved",
    "backfill_no_place",
    "backfill_no_passages",
    "backfill_not_due",
)

# Levels rather than totals: shown as they stood at the end of each span.
LEVELS = ("behind", "waiting_now")


def counts(line: str) -> dict[str, int]:
    return {name: int(value) for name, value in re.findall(r"\b(\w+): (\d+)\b", line)}


def when(line: str) -> datetime | None:
    found = re.match(r"\S*?(\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d)", line)
    return datetime.fromisoformat(found.group(1)) if found else None


def rows(lines, minutes: int):
    """Each span's increase of the watched counts; a restart starts them from nothing again."""
    start_at, start, last = None, None, None
    for line in lines:
        at, seen = when(line), counts(line)
        if at is None or "planned" not in seen:
            continue
        if start is None or (last and seen["planned"] < last["planned"]):
            start_at, start = at, seen
        elif (at - start_at).total_seconds() >= minutes * 60:
            row = {name: seen.get(name, 0) - start.get(name, 0) for name in WATCHED if name in seen}
            yield at, row | {name: seen[name] for name in LEVELS if name in seen}
            start_at, start = at, seen
        last = seen


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--minutes", type=int, default=5)
    args = parser.parse_args()
    shown = None
    for at, row in rows(sys.stdin, args.minutes):
        if shown != list(row):
            shown = list(row)
            print("time    " + " ".join(f"{name:>{max(len(name), 8)}}" for name in shown))
        print(at.strftime("%H:%M   ") + " ".join(f"{row[name]:>{max(len(name), 8)}}" for name in shown))
    return 0


if __name__ == "__main__":
    sys.exit(main())
