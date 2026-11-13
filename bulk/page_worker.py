"""python -m bulk.page_worker: pages read by the shared desearch.extraction for the Rust builders, one framed request at a time; Wikipedia articles the Rust frame cannot cut are framed here."""

from __future__ import annotations

import os

# Answers go out on a private copy of stdout, so a stray print cannot break the framing.
CHANNEL = os.fdopen(os.dup(1), "wb")
os.dup2(2, 1)

import argparse  # noqa: E402
import html  # noqa: E402
import json  # noqa: E402
import struct  # noqa: E402
import sys  # noqa: E402
import traceback  # noqa: E402
from dataclasses import asdict  # noqa: E402
from pathlib import Path  # noqa: E402
from urllib.parse import unquote  # noqa: E402

from lxml import etree  # noqa: E402
from lxml import html as lhtml  # noqa: E402

from desearch.extraction import extract  # noqa: E402

EXTRACT = b"e"
WIKIPEDIA = b"w"
BLOCKED = b"k"
BODY = b"r"
ANCHORS = b"a"
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


def answer(request: bytes) -> dict:
    op, data = request[:1], request[1:]
    if op == EXTRACT:
        (size,) = struct.unpack(">I", data[:4])
        url = data[4 : 4 + size].decode()
        return {"page": asdict(extract(data[4 + size :].decode(), url))}
    if op == WIKIPEDIA:
        article = json.loads(data)
        framed, disambiguation = page_html(article, HIDDEN)
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
    if op == ANCHORS:
        try:
            doc = lhtml.fromstring(data)
        except Exception as error:
            caught = isinstance(error, (ValueError, TypeError))
            return {"raised": {"caught": caught, "repr": repr(error), "type": type(error).__name__}}
        return {"hrefs": [anchor.get("href") for anchor in doc.iter("a")]}
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
    original = f"{image['content_url']}?utm_source={host}&utm_campaign=index&utm_content=original" if image.get("content_url") else None
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
        f'<meta property="og:image" content="{html.escape(shown_image)}">' if shown_image else "",
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
