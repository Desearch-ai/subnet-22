import json
import re
from dataclasses import dataclass, field
from urllib.parse import urljoin

from lxml import etree
from lxml import html as lxml_html

MAX_HTML_CHARS = 8_000_000
MAX_HEADINGS = 30
MAX_HEADING_CHARS = 300
MAX_FIELD_CHARS = 1000
MAX_LD_FALLBACK_CHARS = 256_000

CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
ISO_DATE = re.compile(r"(\d{4})-(\d{2})-(\d{2})")
LD_WRAPPERS = re.compile(r"<!--|-->|//\s*<!\[CDATA\[|//\s*\]\]>|<!\[CDATA\[|\]\]>")
TRAILING_COMMA = re.compile(r",\s*([}\]])")
LD_TYPE = re.compile(r'"@type"\s{0,20}:\s{0,20}(\[[^\[\]{}]{0,500}\]|"[^"]{0,200}")')
LD_TYPE_NAME = re.compile(r'"([^"]{1,200})"')
LANG_TAG = re.compile(r"^[a-z]{2,3}(?:[-_][a-z0-9]{2,8})*$", re.I)
LD_SCRIPTS = etree.XPath(
    "//script[contains(translate(@type, 'LDJSON', 'ldjson'), 'ld+json')]"
)

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


@dataclass
class Structured:
    title: str = ""
    description: str = ""
    lang: str = ""
    canonical: str = ""
    published: str = ""
    author: str = ""
    json_ld_types: list[str] = field(default_factory=list)
    headings: list[str] = field(default_factory=list)
    og_type: str = ""
    article_published: bool = False


def parse(html: str) -> etree._Element | None:
    cleaned = CONTROL_CHARS.sub("", html[:MAX_HTML_CHARS])
    if not cleaned.strip():
        return None
    parser = lxml_html.HTMLParser(
        encoding="utf-8", remove_comments=True, remove_pis=True
    )
    try:
        root = lxml_html.document_fromstring(
            cleaned.encode("utf-8", "replace"), parser=parser
        )
    except (etree.ParserError, ValueError):
        return None
    return root if isinstance(root.tag, str) else None


def squash(text: str | None, limit: int = MAX_FIELD_CHARS) -> str:
    return " ".join((text or "").split())[:limit]


def is_article_type(ld_type: str) -> bool:
    return ld_type.endswith("Article") or ld_type in {
        "BlogPosting",
        "LiveBlogPosting",
        "DiscussionForumPosting",
        "Report",
        "Recipe",
        "HowTo",
    }


def read_structure(root: etree._Element, url: str) -> Structured:
    meta = meta_values(root)
    ld_nodes, ld_types = read_json_ld(root)
    article_nodes = [
        node for node in ld_nodes if any(map(is_article_type, node_types(node)))
    ]
    published_meta = iso_date(meta.get("article:published_time", ""))
    published_ld = first_ld_date(article_nodes)

    return Structured(
        title=squash(first_text(root, "//head/title", "//title"))
        or squash(meta.get("og:title")),
        description=squash(
            meta.get("description")
            or meta.get("og:description")
            or meta.get("twitter:description")
        ),
        lang=read_lang(root, meta),
        canonical=read_canonical(root, url),
        published=published_meta
        or published_ld
        or first_ld_date(ld_nodes)
        or next(
            (d for d in (iso_date(meta.get(k, "")) for k in META_DATE_KEYS) if d), ""
        ),
        author=read_author(article_nodes, ld_nodes, meta),
        json_ld_types=ld_types,
        headings=read_headings(root),
        og_type=squash(meta.get("og:type")).lower(),
        article_published=bool(published_meta or published_ld),
    )


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


def read_headings(root: etree._Element) -> list[str]:
    headings = []
    for node in root.iter("h1", "h2", "h3"):
        text = squash(node.text_content(), MAX_HEADING_CHARS)
        if text:
            headings.append(text)
            if len(headings) == MAX_HEADINGS:
                break
    return headings


def iso_date(value: str) -> str:
    found = ISO_DATE.search(value[:64]) if isinstance(value, str) else None
    if not found:
        return ""
    year, month, day = (int(part) for part in found.groups())
    if 1900 <= year <= 2100 and 1 <= month <= 12 and 1 <= day <= 31:
        return found.group(0)
    return ""


def load_json_ld(raw: str) -> object:
    text = LD_WRAPPERS.sub("", raw).strip()
    for candidate in (text, TRAILING_COMMA.sub(r"\1", text)):
        try:
            return json.loads(candidate, strict=False)
        except (ValueError, RecursionError):
            continue
    return None


def flatten_ld(data: object, depth: int = 0) -> list[dict]:
    if depth > 4:
        return []
    if isinstance(data, list):
        return [node for item in data for node in flatten_ld(item, depth + 1)]
    if not isinstance(data, dict):
        return []
    return [data, *flatten_ld(data.get("@graph"), depth + 1)]


def clean_type(value: str) -> str:
    return value.strip().rsplit("/", 1)[-1].rsplit(":", 1)[-1].strip()


def node_types(node: dict) -> list[str]:
    raw = node.get("@type")
    values = raw if isinstance(raw, list) else [raw]
    return [clean_type(v) for v in values if isinstance(v, str) and clean_type(v)]


def read_json_ld(root: etree._Element) -> tuple[list[dict], list[str]]:
    nodes: list[dict] = []
    types: list[str] = []
    for script in LD_SCRIPTS(root):
        raw = script.text or ""
        data = load_json_ld(raw)
        if data is None:
            if len(raw) <= MAX_LD_FALLBACK_CHARS:
                for found in LD_TYPE.findall(raw):
                    types.extend(clean_type(t) for t in LD_TYPE_NAME.findall(found))
            continue
        for node in flatten_ld(data):
            nodes.append(node)
            types.extend(node_types(node))
    unique = list(dict.fromkeys(t for t in types if t))
    return nodes, unique[:50]


def first_ld_date(nodes: list[dict]) -> str:
    for node in nodes:
        value = node.get("datePublished")
        if isinstance(value, list):
            value = value[0] if value else ""
        date = iso_date(value) if isinstance(value, str) else ""
        if date:
            return date
    return ""


def person_names(value: object, by_id: dict[str, dict]) -> list[str]:
    if isinstance(value, str):
        return [value] if not value.startswith("http") else []
    if isinstance(value, list):
        return [name for item in value[:10] for name in person_names(item, by_id)]
    if isinstance(value, dict):
        name = value.get("name")
        if not name and isinstance(value.get("@id"), str):
            name = by_id.get(value["@id"], {}).get("name")
        return [name] if isinstance(name, str) and name.strip() else []
    return []


def read_author(
    article_nodes: list[dict], ld_nodes: list[dict], meta: dict[str, str]
) -> str:
    by_id = {n["@id"]: n for n in ld_nodes if isinstance(n.get("@id"), str)}
    for node in article_nodes:
        names = list(
            dict.fromkeys(squash(n) for n in person_names(node.get("author"), by_id))
        )
        if names:
            return squash(", ".join(names))
    for key in ("author", "article:author", "dc.creator"):
        value = meta.get(key, "")
        if value and not value.startswith("http"):
            return squash(value)
    return ""
