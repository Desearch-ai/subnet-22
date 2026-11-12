"""python -m bulk.page_worker: pages read by the shared desearch.extraction for the Rust builders, one framed request at a time."""

from __future__ import annotations

import os

# Answers go out on a private copy of stdout, so a stray print cannot break the framing.
CHANNEL = os.fdopen(os.dup(1), "wb")
os.dup2(2, 1)

import argparse  # noqa: E402
import json  # noqa: E402
import struct  # noqa: E402
import sys  # noqa: E402
import traceback  # noqa: E402
from dataclasses import asdict  # noqa: E402
from pathlib import Path  # noqa: E402

from desearch.extraction import extract  # noqa: E402

EXTRACT = b"e"
WIKIPEDIA = b"w"
BLOCKED = b"k"
BODY = b"r"
HIDDEN: set[str] = set()


def answer(request: bytes) -> dict:
    op, data = request[:1], request[1:]
    if op == EXTRACT:
        (size,) = struct.unpack(">I", data[:4])
        url = data[4 : 4 + size].decode()
        return {"page": asdict(extract(data[4 + size :].decode(), url))}
    if op == WIKIPEDIA:
        from bulk import wikipedia

        article = json.loads(data)
        framed, disambiguation = wikipedia.page_html(article, HIDDEN)
        if disambiguation:
            return {"disambiguation": True}
        return {"page": asdict(extract(framed, article["url"]))}
    if op == BLOCKED:
        from desearch.extraction.blocked import looks_blocked

        found = json.loads(data)
        return {"blocked": looks_blocked(found["status"], found["html"], found["text"], found["title"])}
    if op == BODY:
        from desearch.extraction.structured.html import parse
        from desearch.extraction.text.markdown import render

        root = parse(f"<html><body>{data.decode()}</body></html>")
        body = root.find(".//body") if root is not None else None
        return {"lines": None if body is None else [list(line) for line in render(body)]}
    raise ValueError(f"unknown request {op!r}")


def reply(request: bytes) -> bytes:
    try:
        return json.dumps(answer(request), ensure_ascii=False).encode()
    except Exception:
        return json.dumps({"error": "crash", "detail": traceback.format_exc()}).encode()


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m bulk.page_worker")
    parser.add_argument(
        "--hidden",
        type=Path,
        help="hidden category names, one per line, for framing Wikipedia articles",
    )
    args = parser.parse_args()
    if args.hidden:
        HIDDEN.update(args.hidden.read_text().splitlines())
    source, sink = sys.stdin.buffer, CHANNEL
    try:
        while header := source.read(4):
            (size,) = struct.unpack(">I", header)
            data = reply(source.read(size))
            sink.write(struct.pack(">I", len(data)) + data)
            sink.flush()
    except BrokenPipeError:
        # The builder stopped reading, as it does once --limit is reached.
        os._exit(0)


if __name__ == "__main__":
    main()
