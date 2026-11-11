"""python -m bulk.pmc_render: the abstracts and body of PMC articles rendered as `article_page` renders them, for the Rust builder."""

from __future__ import annotations

import json
import os
import struct
import sys
import traceback

from lxml import etree

from bulk.pmc import as_html, render

ABSTRACT_TYPES = (None, "summary", "structured")


def parts(xml: bytes) -> dict:
    try:
        root = etree.fromstring(xml)
    except etree.XMLSyntaxError:
        return {"error": "xml"}
    front = root.find("front/article-meta")
    if front is None:
        return {"abstracts": [], "body": None}
    try:
        abstracts = [
            render(as_html(a, depth=3))
            for a in front.iterfind("abstract")
            if a.get("abstract-type") in ABSTRACT_TYPES
        ]
        body = root.find("body")
        return {
            "abstracts": abstracts,
            "body": render(as_html(body)) if body is not None else None,
        }
    except etree.XMLSyntaxError:
        return {"error": "render"}


def answer(xml: bytes) -> bytes:
    try:
        return json.dumps(parts(xml), ensure_ascii=False).encode()
    except Exception:
        return json.dumps({"error": "crash", "detail": traceback.format_exc()}).encode()


def main() -> None:
    source, sink = sys.stdin.buffer, sys.stdout.buffer
    try:
        while header := source.read(4):
            (size,) = struct.unpack(">I", header)
            data = answer(source.read(size))
            sink.write(struct.pack(">I", len(data)) + data)
            sink.flush()
    except BrokenPipeError:
        # The builder stopped reading, as it does once --limit is reached.
        os._exit(0)


if __name__ == "__main__":
    main()
