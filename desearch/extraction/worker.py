"""python -m desearch.extraction.worker: the shared extraction for the Rust builders and desearch-bot, one length-framed request at a time."""

from __future__ import annotations

import argparse
import base64
import csv
import html
import io
import json
import logging
import os
import re
import signal
import struct
import sys
import traceback
from contextlib import contextmanager
from dataclasses import asdict, replace
from pathlib import Path
from urllib.parse import unquote, urlsplit

from lxml import etree
from lxml import html as lhtml

import desearch.extraction as x
from desearch.extraction.text.lines import clean
from desearch.extraction.text.markdown import Line, finish, render

EXTRACT = b"e"
WIKIPEDIA = b"w"
BLOCKED = b"k"
BODY = b"r"
ANCHORS = b"a"
CRAWLED = b"c"
DOCUMENT = b"d"
PMC = b"p"
HIDDEN: set[str] = set()

PROJECTS = {
    "enwiki": "Wikipedia",
    "enwikisource": "Wikisource",
    "enwikibooks": "Wikibooks",
    "enwikiquote": "Wikiquote",
    "enwikivoyage": "Wikivoyage",
    "enwikinews": "Wikinews",
}
CATEGORY_LINK = "mw:PageProp/Category"
DISAMBIGUATION_MARK = "mw:PageProp/disambiguation"
OG_IMAGE_MAX = 1200
THUMB_STEPS = (20, 40, 60, 120, 250, 330, 500, 960, 1280)
PUBLISHER = {
    "@type": "Organization",
    "name": "Wikimedia Foundation, Inc.",
    "logo": {
        "@type": "ImageObject",
        "url": "https://www.wikimedia.org/static/images/wmf-hor-googpub.png",
    },
}

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


def answer(request: bytes) -> dict:
    op, data = request[:1], request[1:]
    if op == EXTRACT:
        (size,) = struct.unpack(">I", data[:4])
        url = data[4 : 4 + size].decode()
        return {"page": asdict(x.extract(data[4 + size :].decode(), url))}
    if op == WIKIPEDIA:
        article = json.loads(data)
        framed, disambiguation = page_html(article, HIDDEN)
        if disambiguation:
            return {"disambiguation": True}
        return {"page": asdict(x.extract(framed, article["url"]))}
    if op == BLOCKED:
        found = json.loads(data)
        return {
            "blocked": x.looks_blocked(
                found["status"], found["html"], found["text"], found["title"]
            )
        }
    if op == BODY:
        root = x.parse(f"<html><body>{data.decode()}</body></html>")
        body = root.find(".//body") if root is not None else None
        return {
            "lines": None if body is None else [list(line) for line in render(body)]
        }
    if op == ANCHORS:
        try:
            doc = lhtml.fromstring(data)
        except Exception as error:
            caught = isinstance(error, (ValueError, TypeError))
            return {
                "raised": {
                    "caught": caught,
                    "repr": repr(error),
                    "type": type(error).__name__,
                }
            }
        return {"hrefs": [anchor.get("href") for anchor in doc.iter("a")]}
    if op == CRAWLED:
        return crawled_page(json.loads(data))
    if op == DOCUMENT:
        page, which = document_page(json.loads(data))
        return {"page": asdict(page) if page is not None else None, "which": which}
    if op == PMC:
        return article_parts(data)
    raise ValueError(f"unknown request {op!r}")


def category_name(href: str) -> str:
    return unquote(href.split("Category:", 1)[1].split("#", 1)[0]).replace("_", " ")


def page_html(article: dict, hidden: set[str]) -> tuple[str, bool]:
    """The page as the live site frames this revision's Parsoid HTML, and whether it is a disambiguation page."""
    body_html = (article.get("article_body") or {}).get("html") or ""
    root = lhtml.document_fromstring(body_html.encode("utf-8"))
    head_marks = {m.get("property") for m in root.iter("meta")}
    categories = list(
        dict.fromkeys(
            category_name(link.get("href", ""))
            for link in root.iter("link")
            if link.get("rel") == CATEGORY_LINK and "Category:" in link.get("href", "")
        )
    )
    body = root.find("body")
    inner = (
        "".join(
            etree.tostring(child, encoding="unicode", method="html") for child in body
        )
        if body is not None
        else ""
    )
    name = article["name"]
    url = article["url"]
    site = PROJECTS.get(
        (article.get("is_part_of") or {}).get("identifier", ""), "Wikipedia"
    )
    entity = (article.get("main_entity") or {}).get("identifier")
    host = url.split("/")[2]
    image = article.get("image") or {}
    original = (
        f"{image['content_url']}?utm_source={host}&utm_campaign=index&utm_content=original"
        if image.get("content_url")
        else None
    )
    shown_image = og_image(image, host)
    ld = {
        "@context": "https://schema.org",
        "@type": "Article",
        "name": name,
        "url": url,
    }
    if entity:
        ld["sameAs"] = ld["mainEntity"] = f"http://www.wikidata.org/entity/{entity}"
    ld["author"] = {
        "@type": "Organization",
        "name": "Contributors to Wikimedia projects",
    }
    ld["publisher"] = PUBLISHER
    ld.update(
        {
            k: v
            for k, v in (
                ("datePublished", article.get("date_created")),
                ("dateModified", article.get("date_modified")),
                ("image", original),
                ("headline", article.get("description")),
            )
            if v
        }
    )
    robots = (
        "noindex,nofollow,max-image-preview:standard"
        if (article.get("version") or {}).get("noindex")
        else "max-image-preview:standard"
    )
    title = html.escape(f"{name} - {site}")
    head = [
        '<meta charset="UTF-8">',
        f"<title>{title}</title>",
        '<meta name="generator" content="MediaWiki">',
        f'<meta name="robots" content="{robots}">',
        f'<meta property="og:image" content="{html.escape(shown_image)}">'
        if shown_image
        else "",
        f'<meta property="og:title" content="{title}">',
        '<meta property="og:type" content="website">',
        f'<link rel="canonical" href="{html.escape(url)}">',
        f'<script type="application/ld+json">{json.dumps(ld, ensure_ascii=False)}</script>',
    ]
    shown = "".join(
        f'<li><a href="/wiki/Category:{html.escape(c)}">{html.escape(c)}</a></li>'
        for c in categories
        if c not in hidden
    )
    lang = (article.get("in_language") or {}).get("identifier", "en")
    page = (
        f'<!DOCTYPE html><html lang="{lang}" dir="ltr"><head>{"".join(head)}</head><body>'
        f'<div id="mw-content-text" class="mw-body-content"><div class="mw-content-ltr mw-parser-output" lang="{lang}" dir="ltr">{inner}</div></div>'
        f'<div id="catlinks"><div id="mw-normal-catlinks" class="mw-normal-catlinks"><ul>{shown}</ul></div></div></body></html>'
    )
    return page, DISAMBIGUATION_MARK in head_marks


def og_image(image: dict, host: str) -> str:
    """The page image as MediaWiki's og:image shows it: the file itself when it fits 1200 px, else the thumbnail step that holds it scaled to fit."""
    url = image.get("content_url") or ""
    width, height = image.get("width") or 0, image.get("height") or 0
    if "/wikipedia/" not in url or not (width and height):
        return url
    tracking = f"?utm_source={host}&utm_campaign=index&utm_content="
    vector = url.lower().endswith(".svg")
    if not vector and width <= OG_IMAGE_MAX and height <= OG_IMAGE_MAX:
        return f"{url}{tracking}thumbnail_unscaled"
    scaled = round(width * min(OG_IMAGE_MAX / width, OG_IMAGE_MAX / height))
    step = next((s for s in THUMB_STEPS if s >= scaled), THUMB_STEPS[-1])
    project, file_path = url.split("/wikipedia/", 1)[1].split("/", 1)
    name = file_path.rsplit("/", 1)[1]
    return f"https://thumb.wikimedia.org/wikipedia/{project}/thumb/{file_path}/{step}px-{name}{'.png' if vector else ''}{tracking}thumbnail"


def crawled_page(request: dict) -> dict:
    """A crawled HTML page as desearch-bot builds it: EDGAR documents rewritten, and checked for blocks as fetched."""
    markup = request["html"]
    page = edgar_page(
        markup,
        request["url"],
        x.extract(markup, request["url"]),
        request.get("whole", False),
    )
    blocked = x.looks_blocked(200, request.get("orig") or markup, page.text, page.title)
    return {"page": asdict(page), "blocked": bool(blocked)}


def edgar_page(markup: str, url: str, page, whole: bool):
    if not EDGAR_DOCUMENT.match(url) or (page.page_type != x.LISTING and not whole):
        return page
    root = x.parse(markup)
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


def document_page(rec: dict):
    """A fetched PDF, table or data file as a page, or None and why not; the kind it was read as otherwise."""
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


def article_parts(xml: bytes) -> dict:
    """A PMC article's abstracts and body as rendered lines, or the error that stopped them."""
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


def reply(request: bytes) -> bytes:
    try:
        return json.dumps(answer(request), ensure_ascii=False).encode()
    except Exception:
        return json.dumps({"error": "crash", "detail": traceback.format_exc()}).encode()


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m desearch.extraction.worker")
    parser.add_argument(
        "--hidden",
        type=Path,
        help="hidden category names, one per line, for framing Wikipedia articles",
    )
    args = parser.parse_args()
    if args.hidden:
        HIDDEN.update(args.hidden.read_text().splitlines())
    # Answers go out on a private copy of stdout, so a stray print cannot break the framing.
    sink = os.fdopen(os.dup(1), "wb")
    os.dup2(2, 1)
    source = sys.stdin.buffer
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
