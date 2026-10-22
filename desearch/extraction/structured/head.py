"""Fields from the page's head and meta tags: language, canonical, site name, image, section, tags, robots."""

import re
from urllib.parse import urljoin

from lxml import etree

from desearch.extraction.structured.html import MAX_FIELD_CHARS, absolute_url, squash
from desearch.extraction.structured.jsonld import (
    MAX_LABEL_CHARS,
    MAX_NAME_CHARS,
    node_types,
    person_names,
    plain,
    strings,
)

MAX_TAGS = 20
MAX_ROBOTS = 10
LANG_TAG = re.compile(r"^[a-z]{2,3}(?:[-_][a-z0-9]{2,8})*$", re.I)
IMAGE_KEYS = ("og:image", "og:image:url", "og:image:secure_url", "twitter:image")
ROBOTS_NAMES = ("robots", "googlebot")


def meta_values(root: etree._Element) -> dict[str, str]:
    values: dict[str, str] = {}
    for tag in root.iter("meta"):
        key = (
            tag.get("property")
            or tag.get("name")
            or tag.get("itemprop")
            or tag.get("http-equiv")
            or ""
        )
        content = (tag.get("content") or "").strip()
        key = key.strip().lower()
        if key and content and key not in values:
            values[key] = content
    return values


def meta_all(root: etree._Element, key: str) -> list[str]:
    found = []
    for tag in root.iter("meta"):
        name = (tag.get("property") or tag.get("name") or "").strip().lower()
        content = (tag.get("content") or "").strip()
        if name == key and content:
            found.append(content)
    return found


def first_text(root: etree._Element, *paths: str) -> str:
    for path in paths:
        for node in root.xpath(path):
            text = node.text_content()
            if text and text.strip():
                return text
    return ""


def read_lang(root: etree._Element, meta: dict[str, str]) -> str:
    candidates = (
        root.get("lang"),
        root.get("{http://www.w3.org/XML/1998/namespace}lang"),
        meta.get("content-language"),
        meta.get("og:locale"),
    )
    for value in candidates:
        value = (value or "").strip().split(",")[0].strip()
        if LANG_TAG.match(value):
            return re.split(r"[-_]", value)[0].lower()
    return ""


def read_canonical(root: etree._Element, url: str) -> str:
    for link in root.iter("link"):
        rel = (link.get("rel") or "").lower().split()
        href = (link.get("href") or "").strip()
        if "canonical" in rel and href:
            try:
                return urljoin(url or "", href)[:MAX_FIELD_CHARS]
            except ValueError:
                return href[:MAX_FIELD_CHARS]
    return ""


def read_author(
    article_nodes: list[dict], ld_nodes: list[dict], meta: dict[str, str]
) -> str:
    by_id = {n["@id"]: n for n in ld_nodes if isinstance(n.get("@id"), str)}
    for node in article_nodes:
        names = list(
            dict.fromkeys(squash(n) for n in person_names(node.get("author"), by_id))
        )
        if names:
            return squash(", ".join(names[:10]), MAX_NAME_CHARS)
    for key in ("author", "article:author", "dc.creator"):
        value = meta.get(key, "")
        if value and not value.startswith("http"):
            return squash(value, MAX_NAME_CHARS)
    return ""


def read_site_name(meta: dict[str, str], nodes: list[dict]) -> str:
    if meta.get("og:site_name"):
        return squash(meta["og:site_name"], MAX_NAME_CHARS)
    for wanted in ("WebSite", "Organization"):
        for node in nodes:
            if wanted in node_types(node) and plain(node.get("name")):
                return plain(node.get("name"), MAX_NAME_CHARS)
    return ""


def read_image(meta: dict[str, str], url: str) -> str:
    return next(
        (u for u in (absolute_url(meta.get(k), url) for k in IMAGE_KEYS) if u), ""
    )


def read_section(meta: dict[str, str], article_nodes: list[dict]) -> str:
    if meta.get("article:section"):
        return squash(meta["article:section"], MAX_LABEL_CHARS)
    return next(
        (
            s
            for s in (
                plain(n.get("articleSection"), MAX_LABEL_CHARS) for n in article_nodes
            )
            if s
        ),
        "",
    )


def read_tags(
    root: etree._Element, meta: dict[str, str], article_nodes: list[dict]
) -> list[str]:
    found = list(meta_all(root, "article:tag"))
    found += (meta.get("news_keywords") or "").split(",")
    for node in article_nodes:
        keywords = node.get("keywords")
        found += (
            keywords.split(",")
            if isinstance(keywords, str)
            else strings(keywords, MAX_LABEL_CHARS, MAX_TAGS)
        )
    tags = (squash(tag, MAX_LABEL_CHARS) for tag in found)
    return list(dict.fromkeys(tag for tag in tags if tag))[:MAX_TAGS]


def read_robots(root: etree._Element) -> list[str]:
    found = []
    for tag in root.iter("meta"):
        if (tag.get("name") or "").strip().lower() in ROBOTS_NAMES:
            found += (
                part.strip().lower() for part in (tag.get("content") or "").split(",")
            )
    directives = (squash(d, 40) for d in found)
    return list(dict.fromkeys(d for d in directives if d))[:MAX_ROBOTS]
