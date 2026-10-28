"""Wikimedia Enterprise snapshots: each article's Parsoid HTML in the live page's frame, read by the same v2 extraction a crawl uses."""

from __future__ import annotations

import gzip
import html
import json
import tarfile
from collections import Counter
from collections.abc import Iterable, Iterator
from datetime import datetime, timedelta
from pathlib import Path
from urllib.parse import unquote

from lxml import etree
from lxml import html as lhtml

from bulk.rows import Batches, upload_row
from desearch.extraction import extract

PROJECTS = {
    "enwiki": "Wikipedia",
    "enwikisource": "Wikisource",
    "enwikibooks": "Wikibooks",
    "enwikiquote": "Wikiquote",
    "enwikivoyage": "Wikivoyage",
    "enwikinews": "Wikinews",
}
CATEGORY_LINK = "mw:PageProp/Category"
HIDDEN_MARK = "mw:PageProp/hiddencat"
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


def articles(path: Path) -> Iterator[dict]:
    """Every article line of a snapshot: a .tar.gz of NDJSON files, or one NDJSON file, gzipped or not."""
    if path.name.endswith(".tar.gz") or path.name.endswith(".tgz"):
        with tarfile.open(path, "r|gz") as bundle:
            for member in bundle:
                source = bundle.extractfile(member) if member.isfile() else None
                if source is not None:
                    yield from (json.loads(line) for line in source if line.strip())
        return
    with gzip.open(path, "rb") if path.suffix == ".gz" else open(path, "rb") as source:
        yield from (json.loads(line) for line in source if line.strip())


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


def hidden_categories(found: Iterable[dict]) -> Iterator[str]:
    """Names of the hidden categories in a category-namespace snapshot."""
    for article in found:
        if HIDDEN_MARK in ((article.get("article_body") or {}).get("html") or ""):
            yield article["name"].split(":", 1)[-1]


def run(
    paths: list[Path],
    batches: Batches,
    as_of: datetime,
    hidden: set[str],
    limit: int = 0,
) -> Counter:
    counts: Counter = Counter()
    # A snapshot can repeat an article; a later, higher revision gets a later second so it wins over the copy written before.
    revisions: dict[int, tuple[int, int]] = {}
    for path in paths:
        for article in articles(path):
            if (article.get("namespace") or {}).get("identifier") != 0:
                counts["other_namespace"] += 1
                continue
            if (article.get("event") or {}).get("type") == "delete":
                counts["deleted"] += 1
                continue
            revision = (article.get("version") or {}).get("identifier") or 0
            seen = revisions.get(article["identifier"])
            if seen is not None and revision <= seen[0]:
                counts["older_copy"] += 1
                continue
            bump = seen[1] + 1 if seen is not None else 0
            revisions[article["identifier"]] = (revision, bump)
            if not (article.get("article_body") or {}).get("html"):
                counts["no_html"] += 1
                continue
            framed, disambiguation = page_html(article, hidden)
            if disambiguation:
                counts["disambiguation"] += 1
                continue
            page = extract(framed, article["url"])
            if not page.text.strip():
                counts["no_text"] += 1
                continue
            batches.add(
                upload_row(
                    article["url"], page, as_of + timedelta(seconds=bump), "text/html"
                )
            )
            counts["pages"] += 1
            if limit and counts["pages"] >= limit:
                return counts
    return counts
