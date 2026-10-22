"""HTML parsed into a tree, with markup the parser misplaces put back, and the text and URL helpers the field readers share."""

import re
from urllib.parse import urljoin

from lxml import etree
from lxml import html as lxml_html

MAX_HTML_CHARS = 8_000_000
MAX_FIELD_CHARS = 1000
MAX_URL_CHARS = 2048

CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
# Page markup that never belongs inside an SVG drawing.
MISNESTED_IN_SVG = etree.XPath(
    ".//div | .//p | .//section | .//article | .//main | .//header | .//footer | .//nav"
    " | .//ul | .//ol | .//table | .//form | .//h1 | .//h2 | .//h3 | .//h4 | .//h5 | .//h6"
)
SVG_TAGS = frozenset(
    (
        "svg",
        "symbol",
        "g",
        "path",
        "defs",
        "use",
        "circle",
        "rect",
        "ellipse",
        "line",
        "polyline",
        "polygon",
        "lineargradient",
        "radialgradient",
        "stop",
        "clippath",
        "mask",
        "pattern",
        "filter",
        "marker",
        "metadata",
        "title",
        "desc",
        "text",
        "tspan",
        "image",
        "switch",
        "foreignobject",
        "fegaussianblur",
        "feoffset",
        "feblend",
        "fecolormatrix",
        "fecomposite",
        "feflood",
        "femerge",
        "femergenode",
        "feturbulence",
        "fedisplacementmap",
    )
)


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
    if not isinstance(root.tag, str):
        return None
    restore_misnested_svgs(root)
    return root


def restore_misnested_svgs(root: etree._Element) -> None:
    """libxml2 can nest the rest of a page inside an SVG sprite it fails to close: the drawing goes, the page's markup stays in place."""
    for svg in [s for s in root.iter("svg") if MISNESTED_IN_SVG(s)]:
        if not any(a is root for a in svg.iterancestors()):
            continue
        for node in reversed(list(svg.iterdescendants())):
            if not isinstance(node.tag, str) or node.tag not in SVG_TAGS:
                continue
            if any(
                isinstance(d.tag, str) and d.tag not in SVG_TAGS
                for d in node.iterdescendants()
            ):
                node.drop_tag()
            else:
                node.drop_tree()
        svg.drop_tag()


def squash(text: str | None, limit: int = MAX_FIELD_CHARS) -> str:
    return " ".join((text or "").split())[:limit]


def absolute_url(value: object, base: str) -> str:
    if not isinstance(value, str) or not value.strip():
        return ""
    try:
        joined = urljoin(base or "", value.strip())
    except ValueError:
        return ""
    return (
        joined
        if joined.startswith(("http://", "https://")) and len(joined) <= MAX_URL_CHARS
        else ""
    )
