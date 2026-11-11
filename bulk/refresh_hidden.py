"""Rewrite one source's section of index-1's hidden-URLs file from the URLs on stdin (stdlib only, runs on index-1)."""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path
from urllib.parse import urlsplit

HIDDEN = Path("/data0/index/hidden-urls.txt")
NAME = re.compile(r"^[a-z0-9][a-z0-9-]{0,40}$")
GLOB = re.compile(r"([*?\[])")


def rule(url: str) -> str | None:
    """The router's `host/path-glob` line matching exactly this URL's path, or None when a rule can't say it."""
    parts = urlsplit(url.strip())
    if not parts.hostname or parts.query or "#" in parts.path or any(c.isspace() for c in parts.path):
        return None
    return parts.hostname + GLOB.sub(r"[\1]", parts.path or "/")


def rewrite(text: str, source: str, urls: list[str]) -> str:
    begin, end = f"# refresh {source} begin", f"# refresh {source} end"
    kept, inside = [], False
    for line in text.splitlines():
        if line == begin:
            inside = True
        elif line == end:
            inside = False
        elif not inside:
            kept.append(line)
    rules = sorted({r for r in map(rule, urls) if r})
    if rules:
        kept += [begin, *rules, end]
    return "\n".join(kept) + "\n"


def main() -> int:
    source = (
        os.environ.get("SSH_ORIGINAL_COMMAND")
        or (sys.argv[1] if len(sys.argv) > 1 else "")
    ).strip()
    if not NAME.match(source):
        print(f"not a source name: {source!r}", file=sys.stderr)
        return 2
    path = Path(sys.argv[2]) if len(sys.argv) > 2 else HIDDEN
    before = path.read_text() if path.exists() else ""
    after = rewrite(before, source, sys.stdin.read().split())
    if after != before:
        tmp = path.with_name(path.name + f".refresh-{source}.tmp")
        tmp.write_text(after)
        os.replace(tmp, path)
    print(f"{source}: {after.count(chr(10)) - before.count(chr(10)):+d} lines")
    return 0


if __name__ == "__main__":
    sys.exit(main())
