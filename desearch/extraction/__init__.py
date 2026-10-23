import re
from dataclasses import dataclass, field, fields, replace

from desearch.extraction.blocked import looks_blocked
from desearch.extraction.classify import (
    ARTICLE,
    DOCS,
    LISTING,
    OTHER,
    PRODUCT,
    classify,
)
from desearch.extraction.structured import read_structure
from desearch.extraction.structured.html import parse
from desearch.extraction.text.article import main_text
from desearch.extraction.text.dom import MAX_TEXT_CHARS
from desearch.extraction.text.mediawiki import is_mediawiki, mediawiki_text
from desearch.extraction.text.page import MAX_LISTING_CHARS, full_text

__all__ = [
    "EXTRACTION_V1",
    "EXTRACTION_V2",
    "FULL",
    "MAIN",
    "Page",
    "extract",
    "looks_blocked",
]

EXTRACTION_V1 = "v1"
EXTRACTION_V2 = "v2"

MAIN = "main"
FULL = "full"
MAIN_TYPES = (ARTICLE, DOCS)
MIN_MAIN_SHARE = 0.4
# Article scoping prunes widgets itself, so a smaller share of the page is still the whole story.
MIN_ARTICLE_MAIN_SHARE = 0.2
MIN_TRUSTED_MAIN_CHARS = 1500
SURROGATES = re.compile("[\ud800-\udfff]")
MAX_HEADINGS = 30
MAX_HEADING_CHARS = 300
HEADING_MARKS = ("# ", "## ", "### ")
FENCE = "```"


@dataclass
class Page:
    page_type: str = OTHER
    text_mode: str = FULL
    title: str = ""
    description: str = ""
    lang: str = ""
    canonical: str = ""
    published: str = ""
    modified: str = ""
    author: str = ""
    site_name: str = ""
    image: str = ""
    section: str = ""
    tags: list[str] = field(default_factory=list)
    breadcrumbs: list[str] = field(default_factory=list)
    json_ld_types: list[str] = field(default_factory=list)
    headings: list[str] = field(default_factory=list)
    entity: str = ""
    ld: str = ""
    paywalled: bool | None = None
    robots: list[str] = field(default_factory=list)
    text: str = ""


def extract(html: str | bytes, url: str) -> Page:
    if isinstance(html, bytes):
        html = html.decode("utf-8", "replace")
    try:
        root = parse(html or "")
        if root is None:
            return Page()
        meta = read_structure(root, url or "")
        page_type = classify(url or "", meta, root)
        text, text_mode, tags = page_text(
            root, url or "", meta.title, meta.generator, page_type, meta.author
        )
    except Exception:
        return Page()

    return without_surrogates(
        Page(
            page_type=page_type,
            text_mode=text_mode,
            title=meta.title,
            description=meta.description,
            lang=meta.lang,
            canonical=meta.canonical,
            published=meta.published,
            modified=meta.modified,
            author=meta.author,
            site_name=meta.site_name,
            image=meta.image,
            section=meta.section,
            tags=meta.tags or tags,
            breadcrumbs=meta.breadcrumbs,
            json_ld_types=meta.json_ld_types,
            headings=text_headings(text),
            entity=meta.entity,
            ld=meta.ld,
            paywalled=meta.paywalled,
            robots=meta.robots,
            text=text,
        )
    )


def without_surrogates(page: Page) -> Page:
    """JSON-LD escapes can decode to lone surrogates, which no UTF-8 file can hold."""
    found = {}
    for name in (f.name for f in fields(Page)):
        value = getattr(page, name)
        if isinstance(value, str) and SURROGATES.search(value):
            found[name] = SURROGATES.sub("", value)
        elif isinstance(value, list) and any(SURROGATES.search(v) for v in value):
            found[name] = [SURROGATES.sub("", v) for v in value]
    return replace(page, **found) if found else page


def page_text(
    root, url: str, title: str, generator: str, page_type: str, author: str = ""
) -> tuple[str, str, list[str]]:
    """The page's text, the mode that made it, and categories found while making it (MediaWiki pages)."""
    if page_type in MAIN_TYPES and is_mediawiki(root, generator):
        text, categories = mediawiki_text(root, title)
        return text, MAIN, categories[:20]
    if page_type not in MAIN_TYPES:
        cap = MAX_LISTING_CHARS if page_type == LISTING else MAX_TEXT_CHARS
        return full_text(root, title, cap, page_type == PRODUCT), FULL, []
    main = main_text(root, title, url, page_type == ARTICLE, author=author)
    if main and len(main) >= MIN_TRUSTED_MAIN_CHARS:
        return main, MAIN, []
    whole = full_text(root, title)
    # A short main text beside a long page means the scope missed the body.
    share = MIN_ARTICLE_MAIN_SHARE if page_type == ARTICLE else MIN_MAIN_SHARE
    if main and len(main) >= share * len(whole):
        return main, MAIN, []
    return whole, FULL, []


def text_headings(text: str) -> list[str]:
    """The text's h1 to h3 lines without their marks, outside fenced code."""
    lines, found, i = text.splitlines(), [], 0
    while i < len(lines) and len(found) < MAX_HEADINGS:
        line = lines[i]
        if line.startswith(FENCE):
            close = next(
                (j for j in range(i + 1, len(lines)) if lines[j].startswith(FENCE)),
                None,
            )
            if close is not None:
                i = close + 1
                continue
        if line.startswith(HEADING_MARKS):
            heading = line.split(" ", 1)[1].strip()[:MAX_HEADING_CHARS]
            if heading:
                found.append(heading)
        i += 1
    return found
