from dataclasses import dataclass, field

from desearch.extraction.blocked import looks_blocked
from desearch.extraction.classify import classify
from desearch.extraction.structured import parse, read_structure
from desearch.extraction.text import main_text

__all__ = ["Page", "extract", "looks_blocked"]


@dataclass
class Page:
    page_type: str = "other"
    title: str = ""
    description: str = ""
    lang: str = ""
    canonical: str = ""
    published: str = ""
    author: str = ""
    json_ld_types: list[str] = field(default_factory=list)
    headings: list[str] = field(default_factory=list)
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
        text = main_text(root, page_type)
    except Exception:
        return Page()

    return Page(
        page_type=page_type,
        title=meta.title,
        description=meta.description,
        lang=meta.lang,
        canonical=meta.canonical,
        published=meta.published,
        author=meta.author,
        json_ld_types=meta.json_ld_types,
        headings=meta.headings,
        text=text,
    )
