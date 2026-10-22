"""The page's type from its URL, structured data and layout; the type decides which text mode it gets."""

import re
from urllib.parse import urlsplit

from lxml import etree

from desearch.extraction.classify.features import Features, page_features
from desearch.extraction.classify.hints import (
    COMPANY_SEGMENTS,
    COMPANY_TYPES,
    CONTENT_PARENTS,
    DOCS_GENERATOR,
    DOCS_HOST_LABELS,
    DOCS_SEGMENTS,
    DOCS_TYPES,
    EVENT_SEGMENTS,
    EVENT_TYPES,
    FAQ_TITLE,
    FORUM_GENERATOR,
    FORUM_PATH,
    FORUM_QUERY,
    FORUM_TYPES,
    LANDING_SEGMENTS,
    LANDING_TYPES,
    LEGAL_SEGMENTS,
    LEGAL_TITLE,
    LISTING_KEYS,
    LISTING_SEGMENTS,
    LISTING_TYPES,
    NEAR_ROOT_LEGAL_SEGMENTS,
    NOT_DOCS_SEGMENTS,
    PRODUCT_OG,
    PRODUCT_TYPES,
    SEARCH_KEYS,
    SEARCH_SEGMENTS,
    SECTION_SEGMENTS,
    VERSION,
    WIKI_MAIN_PAGES,
)
from desearch.extraction.classify.urls import (
    YEAR,
    is_date_archive,
    is_home,
    is_paginated,
    path_segments,
    query_pairs,
    words,
)
from desearch.extraction.structured import Structured
from desearch.extraction.structured.jsonld import is_article_type

HOME = "home"
ARTICLE = "article"
LISTING = "listing"
PRODUCT = "product"
DOCS = "docs"
FORUM = "forum"
COMPANY = "company"
LANDING = "landing"
OTHER = "other"
LISTING_LINK_SHARE = 0.7
FRONT_LINK_SHARE = 0.5


def is_event_type(ld_type: str) -> bool:
    return ld_type.endswith("Event") or ld_type in EVENT_TYPES


def classify(url: str, meta: Structured, root: etree._Element) -> str:
    """The page's type from its URL, structured data and layout; the type decides which text mode it gets."""
    segments = path_segments(url)
    pairs = query_pairs(url)
    searched = any(k in SEARCH_KEYS and v.strip() for k, v in pairs)
    if is_home(segments):
        return LISTING if searched else HOME
    last = segments[-1]
    if last.split(":")[-1] in WIKI_MAIN_PAGES:
        return HOME
    tagged = any(s in LISTING_SEGMENTS for s in segments[:-1])
    if not tagged and (
        last in LEGAL_SEGMENTS
        or (last in NEAR_ROOT_LEGAL_SEGMENTS and len(segments) <= 2)
        or LEGAL_TITLE.match(meta.title)
    ):
        return OTHER

    f = page_features(root)
    types = set(meta.json_ld_types)
    published = meta.article_published
    article_typed = any(map(is_article_type, types))
    listing_typed = bool(types & LISTING_TYPES)
    dense = f.long_count >= 3 and f.long_chars >= 1200
    strong_article = article_typed and (
        f.long_chars >= 1500 if listing_typed else published or f.long_chars >= 600
    )
    path = urlsplit(url).path

    if last in COMPANY_SEGMENTS and not any(s in LISTING_SEGMENTS for s in segments):
        return COMPANY
    forum_typed = bool(types & FORUM_TYPES) or "discussion" in meta.og_type
    forum_path = bool(
        FORUM_PATH.search(path) or FORUM_QUERY.search(urlsplit(url).query)
    )
    if forum_typed or (
        forum_path and (FORUM_GENERATOR.search(meta.generator) or f.time_elements >= 2)
    ):
        return FORUM
    if any(s in SEARCH_SEGMENTS for s in segments) or (searched and len(segments) <= 2):
        return LISTING
    listing_path = (
        any(s in LISTING_SEGMENTS for s in segments)
        or last in SECTION_SEGMENTS
        or is_paginated(url, segments, pairs)
        or is_date_archive(segments)
        or any(k in LISTING_KEYS and v.strip() for k, v in pairs)
    )
    if listing_path and not (strong_article and published):
        return LISTING

    product_typed = bool(types & PRODUCT_TYPES) or f.price_props
    product_og = bool(PRODUCT_OG.match(meta.og_type)) and (
        f.prices >= 1 or f.cart_buttons >= 1
    )
    shop = f.cart_buttons >= 1 and f.prices >= 1
    many = f.cart_buttons >= 4 or listing_typed or "listing" in meta.og_type
    if (
        (product_typed or product_og or shop)
        and not many
        and not (strong_article and published)
    ):
        return PRODUCT
    event = any(map(is_event_type, types)) or any(
        s in EVENT_SEGMENTS for s in segments[:-1]
    )
    # Event pages read like articles under a content section, but their date and venue sit outside the body.
    if event and not (strong_article and published):
        return OTHER
    if strong_article or (meta.og_type == "article" and published):
        return ARTICLE

    if types & COMPANY_TYPES or any(
        s in COMPANY_SEGMENTS or s.startswith("about-") for s in segments[-2:]
    ):
        return COMPANY
    docs_type = docs_kind(url, meta, segments, types, f)
    if docs_type:
        return docs_type
    link_heavy = listing_typed or f.link_share >= LISTING_LINK_SHARE
    if not link_heavy and (
        types & LANDING_TYPES or any(s in LANDING_SEGMENTS for s in segments)
    ):
        return LANDING

    in_content_section = any(s in CONTENT_PARENTS for s in segments[:-1])
    slug_like = sum(w.isalpha() for w in words(last)) >= 4
    if dense and (published or in_content_section or slug_like):
        return ARTICLE
    dated = any(YEAR.match(s) for s in segments[:-1])
    if (
        dated
        and (f.long_chars >= 400 or f.paragraph_chars >= 800)
        and (len(words(last)) >= 2 or re.search(r"\d{3,}", last))
    ):
        return ARTICLE
    if (
        in_content_section
        and f.article_elements <= 2
        and f.long_count >= 2
        and f.long_chars >= 600
        and f.headline_links < 10
    ):
        return ARTICLE
    if link_heavy:
        return LISTING
    if len(segments) == 1 and len(words(segments[0])) <= 3:
        front = (
            f.headline_links >= 8
            or f.article_elements >= 6
            or f.time_elements >= 6
            or f.link_share >= FRONT_LINK_SHARE
        )
        return LISTING if front else LANDING
    if f.article_elements >= 6 or (
        (f.headline_links >= 12 or f.time_elements >= 8) and f.long_chars < 3000
    ):
        return LISTING
    return OTHER


def docs_kind(
    url: str, meta: Structured, segments: list[str], types: set[str], f: Features
) -> str:
    """docs for documentation and help pages, listing for wiki special pages, "" otherwise."""
    if any(s in NOT_DOCS_SEGMENTS for s in segments) or any(
        YEAR.match(s) for s in segments
    ):
        return ""
    host_labels = set((urlsplit(url).hostname or "").lower().split(".")[:-2])
    located = (
        any(s in DOCS_SEGMENTS for s in segments)
        or bool(host_labels & DOCS_HOST_LABELS)
        or any(VERSION.match(s) for s in segments[:-1])
    )
    generated = bool(DOCS_GENERATOR.search(meta.generator))
    faq = "FAQPage" in types and (
        any("faq" in s for s in segments) or bool(FAQ_TITLE.search(meta.title))
    )
    if generated and any(s.startswith("special:") for s in segments):
        return LISTING
    texty = (
        f.docs_markers
        or f.paragraph_chars >= 300
        or f.long_count >= 1
        or f.visible_chars >= 800
    )
    if types & DOCS_TYPES or faq or generated or (located and texty):
        return DOCS
    return ""
