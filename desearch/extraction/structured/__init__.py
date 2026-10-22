"""The page's structured fields, read from its head, meta tags and JSON-LD."""

from dataclasses import dataclass, field

from lxml import etree

from desearch.extraction.structured.dates import read_date
from desearch.extraction.structured.head import (
    first_text,
    meta_values,
    read_author,
    read_canonical,
    read_image,
    read_lang,
    read_robots,
    read_section,
    read_site_name,
    read_tags,
)
from desearch.extraction.structured.html import squash
from desearch.extraction.structured.jsonld import (
    first_ld_date,
    is_article_type,
    node_types,
    read_breadcrumbs,
    read_entity,
    read_json_ld,
    read_ld,
    read_paywalled,
)

CONTENT_TIMES = etree.XPath("//article//time[@datetime] | //main//time[@datetime]")

META_DATE_KEYS = (
    "article:published_time",
    "og:published_time",
    "datepublished",
    "date",
    "pubdate",
    "publishdate",
    "publish-date",
    "dc.date.issued",
    "dc.date",
    "sailthru.date",
)
MODIFIED_KEYS = ("article:modified_time", "og:updated_time")


@dataclass
class Structured:
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
    entity: str = ""
    ld: str = ""
    paywalled: bool | None = None
    robots: list[str] = field(default_factory=list)
    og_type: str = ""
    generator: str = ""
    article_published: bool = False


def read_structure(root: etree._Element, url: str) -> Structured:
    meta = meta_values(root)
    ld_nodes, ld_types = read_json_ld(root)
    article_nodes = [
        node for node in ld_nodes if any(map(is_article_type, node_types(node)))
    ]
    published_meta = read_date(meta.get("article:published_time", ""))
    published_ld = first_ld_date(article_nodes, "datePublished")
    published = (
        published_meta
        or published_ld
        or first_ld_date(ld_nodes, "datePublished")
        or next(
            (d for d in (read_date(meta.get(k, "")) for k in META_DATE_KEYS) if d), ""
        )
        or next(
            (
                d
                for d in (read_date(t.get("datetime")) for t in CONTENT_TIMES(root))
                if d
            ),
            "",
        )
    )
    modified = (
        read_date(meta.get(MODIFIED_KEYS[0], ""))
        or first_ld_date(article_nodes, "dateModified")
        or read_date(meta.get(MODIFIED_KEYS[1], ""))
    )
    if published and modified and modified[:10] < published[:10]:
        modified = ""

    return Structured(
        title=squash(first_text(root, "//head/title", "//title"))
        or squash(meta.get("og:title"))
        or squash(first_text(root, "//h1")),
        description=squash(
            meta.get("description")
            or meta.get("og:description")
            or meta.get("twitter:description")
        ),
        lang=read_lang(root, meta),
        canonical=read_canonical(root, url),
        published=published,
        modified=modified,
        author=read_author(article_nodes, ld_nodes, meta),
        site_name=read_site_name(meta, ld_nodes),
        image=read_image(meta, url),
        section=read_section(meta, article_nodes),
        tags=read_tags(root, meta, article_nodes),
        breadcrumbs=read_breadcrumbs(ld_nodes),
        json_ld_types=ld_types,
        entity=read_entity(ld_nodes),
        ld=read_ld(ld_nodes, article_nodes),
        paywalled=read_paywalled(article_nodes or ld_nodes),
        robots=read_robots(root),
        og_type=squash(meta.get("og:type")).lower(),
        generator=squash(meta.get("generator")),
        article_published=bool(published_meta or published_ld),
    )
