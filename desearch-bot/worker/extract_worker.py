"""The extraction worker of `desearch-bot build` and `refresh`: one JSON request a line on stdin, one JSON answer a line on stdout."""

from __future__ import annotations

import argparse
import base64
import csv
import io
import json
import logging
import re
import signal
import sys
from contextlib import contextmanager
from dataclasses import asdict, replace
from urllib.parse import unquote, urlsplit

EDGAR_DOCUMENT = re.compile(
    r"^https?://(www\.)?sec\.gov/Archives/edgar/data/\d+/\d+/(?!R\d+\.htm)[^/]+\.(htm|html|txt)$",
    re.I,
)
EDGAR_TEXT_CHARS = 2_000_000
MIN_PDF_CHARS = 200
MAX_DATA_BYTES = 200_000
MAX_PDF_PAGES = 400
PDF_SECONDS = 30
MAX_TABLE_ROWS = 5_000
CONTENT_TYPES = {
    "application/pdf": "pdf",
    "application/x-pdf": "pdf",
    "text/csv": "csv",
    "application/csv": "csv",
    "application/json": "json",
    "text/json": "json",
    "text/plain": "txt",
    "application/vnd.ms-excel": "xls",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": "xlsx",
}
EXTENSIONS = {"pdf", "csv", "json", "jsonl", "txt", "tsv", "xls", "xlsx"}
POOR_TITLE = re.compile(
    r"^(untitled|microsoft word - .*|document\d*|slide \d+|\W*)$", re.I
)
logging.getLogger("pypdf").setLevel(logging.ERROR)


class Overtime(Exception):
    pass


@contextmanager
def time_limit(seconds: float):
    def stop(*_):
        raise Overtime

    before = signal.signal(signal.SIGALRM, stop)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, before)


def edgar_page(x, html: str, url: str, page, whole: bool):
    if not EDGAR_DOCUMENT.match(url) or (page.page_type != x.LISTING and not whole):
        return page
    root = x.parse(html)
    if root is None:
        return page
    text = x.full_text(
        root, page.title, EDGAR_TEXT_CHARS if whole else x.MAX_TEXT_CHARS, False
    )
    return x.without_surrogates(
        replace(
            page,
            page_type=x.OTHER,
            text_mode=x.FULL,
            text=text,
            headings=x.text_headings(text),
        )
    )


def kind(url: str, content_type: str, body: bytes) -> str | None:
    if body[:5] == b"%PDF-":
        return "pdf"
    if content_type in CONTENT_TYPES:
        return CONTENT_TYPES[content_type]
    ext = urlsplit(url).path.rsplit(".", 1)[-1].lower()
    return (
        ext
        if ext in EXTENSIONS and content_type in ("", "application/octet-stream")
        else None
    )


def file_name(url: str) -> str:
    name = unquote(urlsplit(url).path.rstrip("/").rsplit("/", 1)[-1])
    return (
        re.sub(r"[_-]+", " ", name.rsplit(".", 1)[0]).strip()
        or urlsplit(url).hostname
        or url
    )


def pdf_lines(body: bytes, url: str):
    try:
        with time_limit(PDF_SECONDS):
            return _pdf_lines(body, url)
    except Overtime:
        return "pdf_overtime"


def _pdf_lines(body: bytes, url: str):
    from pypdf import PdfReader

    try:
        reader = PdfReader(io.BytesIO(body))
        if reader.is_encrypted and not reader.decrypt(""):
            return "encrypted"
        pages = [(page.extract_text() or "") for page in reader.pages[:MAX_PDF_PAGES]]
        meta_title = ((reader.metadata or {}).get("/Title") or "").strip()
    except Overtime:
        raise
    except Exception:
        return "unreadable"
    if sum(len(p.strip()) for p in pages) < MIN_PDF_CHARS:
        return "no_text_layer"
    lines: list[str] = []
    for n, text in enumerate(pages, 1):
        body_lines = [re.sub(r"\s+", " ", line).strip() for line in text.splitlines()]
        body_lines = [line for line in body_lines if line]
        if body_lines:
            lines += [f"## Page {n}", *body_lines]
    first = next((line for line in lines if not line.startswith("## ")), "")
    title = (
        meta_title if meta_title and not POOR_TITLE.match(meta_title) else first[:200]
    )
    return title or file_name(url), lines


def table_lines(text: str, delimiter: str) -> list[str]:
    out = []
    for n, row in enumerate(csv.reader(io.StringIO(text), delimiter=delimiter)):
        if n >= MAX_TABLE_ROWS:
            break
        cells = [c.strip() for c in row]
        if any(cells):
            out.append(" | ".join(cells))
    return out


def data_lines(body: bytes, which: str):
    if which in ("xls", "xlsx"):
        return "spreadsheet"
    if len(body) > MAX_DATA_BYTES:
        return "data_too_large"
    text = body.decode("utf-8", errors="replace")
    if which in ("csv", "tsv"):
        return table_lines(text, "\t" if which == "tsv" else ",")
    if which in ("json", "jsonl"):
        try:
            docs = (
                [json.loads(line) for line in text.splitlines() if line.strip()]
                if which == "jsonl"
                else [json.loads(text)]
            )
        except ValueError:
            return "unreadable"
        return [
            line
            for d in docs
            for line in json.dumps(d, ensure_ascii=False, indent=1).splitlines()
        ]
    return [line.strip() for line in text.splitlines() if line.strip()]


def document_page(x, rec: dict):
    from desearch.extraction.text.lines import clean
    from desearch.extraction.text.markdown import Line, finish

    body = base64.b64decode(rec["body_b64"])
    which = rec.get("document") or kind(rec["url"], rec.get("content_type", ""), body)
    if which == "pdf":
        got = pdf_lines(body, rec["url"])
        if isinstance(got, str):
            return None, got
        title, lines = got
    elif which:
        lines = data_lines(body, which)
        if isinstance(lines, str):
            return None, lines
        title = file_name(rec["url"])
    else:
        return None, "not_a_document"
    if not lines:
        return None, "empty"
    text = finish(
        title,
        clean([Line(line, row=" | " in line) for line in lines]),
        x.MAX_TEXT_CHARS,
    )
    page = x.Page(
        text_mode=x.MAIN,
        title=title,
        canonical=rec["url"],
        headings=x.text_headings(text),
        text=text,
    )
    return x.without_surrogates(page), which


def answer(x, req: dict) -> dict:
    if req["op"] == "doc":
        page, which = document_page(x, req)
        return {"page": asdict(page) if page is not None else None, "which": which}
    html = req["html"]
    page = edgar_page(
        x, html, req["url"], x.extract(html, req["url"]), req.get("whole", False)
    )
    blocked = x.looks_blocked(200, req.get("orig") or html, page.text, page.title)
    return {"page": asdict(page), "blocked": bool(blocked)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--subnet22", required=True, help="a subnet-22 checkout with desearch/"
    )
    args = ap.parse_args()
    sys.path.insert(0, args.subnet22)
    import desearch.extraction as x

    out = sys.stdout.buffer
    for line in sys.stdin.buffer:
        try:
            got = answer(x, json.loads(line))
        except Exception as e:
            got = {"error": f"{type(e).__name__}: {e}"}
        out.write(json.dumps(got).encode("ascii") + b"\n")
        out.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main())
