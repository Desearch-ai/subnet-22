"""python -m bulk.pmc_render: the abstracts and body of PMC articles as HTML render() reads, rendered for the Rust builder."""

from __future__ import annotations

import os

# Answers go out on a private copy of stdout, so a stray print cannot break the framing.
CHANNEL = os.fdopen(os.dup(1), "wb")
os.dup2(2, 1)

import json  # noqa: E402
import re  # noqa: E402
import struct  # noqa: E402
import sys  # noqa: E402
import traceback  # noqa: E402

from lxml import etree  # noqa: E402
from lxml import html as lhtml  # noqa: E402

from desearch.extraction.text.markdown import render  # noqa: E402

ABSTRACT_TYPES = (None, "summary", "structured")
SEPARATOR = re.compile(r"[\s,;–—-]*")
EMPTY_BRACKETS = re.compile(r"\s*(\[[\s,;–-]*\]|\([\s,;–-]*\))")
DROPPED = (
    "label",
    "disp-formula",
    "inline-formula",
    "supplementary-material",
    "fn-group",
    "alternatives",
    "ref-list",
    "graphic",
    "inline-graphic",
    "media",
    "table-wrap-foot",
    "object-id",
)
BLOCK_TAGS = {
    "sec": "div",
    "p": "p",
    "list": "ul",
    "list-item": "li",
    "table-wrap": "div",
    "fig": "div",
    "caption": "p",
    "boxed-text": "div",
    "statement": "div",
    "def-list": "ul",
    "def-item": "li",
    "disp-quote": "blockquote",
    "verse-group": "div",
    "preformat": "pre",
    "app": "div",
    "app-group": "div",
}
INLINE_TAGS = {"italic": "i", "bold": "b", "sup": "sup", "sub": "sub", "term": "b"}


def local_name(node: etree._Element) -> str:
    return etree.QName(node).localname if isinstance(node.tag, str) else ""


def as_html(node: etree._Element, depth: int = 2) -> etree._Element:
    """A JATS subtree with HTML tags render() reads: sections as divs led by `h2`-`h4` titles, citation marks and formulas dropped."""
    tree = etree.fromstring(etree.tostring(node))
    drop_citations(tree)
    for drop in [e for e in tree.iter() if dropped(e, tree)]:
        if drop.getparent() is not None:
            drop_keep_tail(drop)
    tags = [
        (e, html_tag(e, tree, depth)) for e in tree.iter() if isinstance(e.tag, str)
    ]
    for element, tag in tags:
        element.tag = tag
        element.attrib.clear()
    etree.cleanup_namespaces(tree)
    return lhtml.fromstring(etree.tostring(tree, encoding="unicode"))


def is_citation(element: etree._Element | None) -> bool:
    return (
        element is not None
        and local_name(element) == "xref"
        and element.get("ref-type") == "bibr"
    )


def drop_citations(tree: etree._Element) -> None:
    """Citation marks out, with the commas and dashes between them, then any brackets or superscripts left empty."""
    for xref in [e for e in tree.iter() if is_citation(e)]:
        if is_citation(xref.getnext()) and SEPARATOR.fullmatch(xref.tail or ""):
            xref.tail = ""
        drop_keep_tail(xref)
    for mark in [
        e
        for e in tree.iter()
        if local_name(e) in ("sup", "sub")
        and len(e) == 0
        and SEPARATOR.fullmatch(e.text or "")
    ]:
        drop_keep_tail(mark)
    for element in tree.iter():
        if isinstance(element.tag, str):
            element.text = element.text and EMPTY_BRACKETS.sub("", element.text)
            element.tail = element.tail and EMPTY_BRACKETS.sub("", element.tail)


def dropped(element: etree._Element, tree: etree._Element) -> bool:
    name = local_name(element)
    if (
        name == "title"
        and element.getparent() is tree
        and local_name(tree) == "abstract"
    ):
        return True
    return name in DROPPED


def html_tag(element: etree._Element, tree: etree._Element, depth: int) -> str:
    name = local_name(element)
    if element is tree:
        return "div"
    if name == "title" and local_name(element.getparent()) in (
        "sec",
        "app",
        "ack",
        "boxed-text",
    ):
        sections = sum(1 for a in element.iterancestors() if local_name(a) == "sec")
        return f"h{max(depth, min(4, depth + sections - 1))}"
    if name == "list" and element.get("list-type") == "order":
        return "ol"
    if name in BLOCK_TAGS:
        return BLOCK_TAGS[name]
    if name in INLINE_TAGS:
        return INLINE_TAGS[name]
    if name in ("table", "thead", "tbody", "tr", "th", "td"):
        return name
    return "span"


def drop_keep_tail(node: etree._Element) -> None:
    parent, tail = node.getparent(), node.tail or ""
    previous = node.getprevious()
    if previous is not None:
        previous.tail = (previous.tail or "") + tail
    else:
        parent.text = (parent.text or "") + tail
    parent.remove(node)


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
    source, sink = sys.stdin.buffer, CHANNEL
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
