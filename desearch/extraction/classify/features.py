"""Layout features of a page: long paragraphs, link share, headline links, prices, cart buttons, docs markers."""

import re
from dataclasses import dataclass

from lxml import etree

LONG_PARAGRAPH = 120
MAX_BODY_CHARS = 200_000
DOCS_MARKERS = re.compile(
    r"\b(on this page|edit this page|edit on github|was this (page|article) helpful|table of contents"
    r"|in this article|improve this page|view source|last updated on|next topic|previous topic|search docs"
    r"|search documentation|documentation home)\b",
    re.I,
)
CART = re.compile(
    r"\b(add to (cart|bag|basket|trolley)|buy (it )?now|add to wish ?list)\b", re.I
)
PRICE = re.compile(
    r"(?:[$£€¥₹]\s?\d[\d,]*(?:\.\d{2})?|\b\d[\d,]*(?:\.\d{2})\s?(?:USD|EUR|GBP|CAD|AUD)\b)"
)

DROPPED = (
    "script",
    "style",
    "noscript",
    "template",
    "svg",
    "canvas",
    "iframe",
    "object",
    "embed",
    "select",
)
CHROME = ("nav", "header", "footer", "aside")
CHROME_ROLES = ("navigation", "banner", "contentinfo")
UNLINKED_PARAGRAPHS = etree.XPath(
    "//p[not(ancestor::nav or ancestor::header or ancestor::footer or ancestor::aside or ancestor::a)]"
)
CHROME_PARAGRAPHS = etree.XPath(
    "//p[not(ancestor::nav or ancestor::header or ancestor::footer or ancestor::aside)]"
)
HEADLINE_LINKS = etree.XPath(
    "count(//h2//a | //h3//a | //h4//a | //a[.//h2 or .//h3 or .//h4])"
)
ARTICLE_ELEMENTS = etree.XPath("count(//article)")
TIME_ELEMENTS = etree.XPath("count(//time)")
PRICE_PROPS = etree.XPath("//*[@itemprop='price' or @itemprop='offers']")
BUTTONS = etree.XPath("//button | //input[@type='submit'] | //*[@role='button']")


@dataclass
class Features:
    long_count: int
    long_chars: int
    paragraph_chars: int
    link_share: float
    visible_chars: int
    headline_links: int
    article_elements: int
    time_elements: int
    price_props: bool
    prices: int
    cart_buttons: int
    docs_markers: int


def visible_link_share(root: etree._Element) -> tuple[float, int]:
    """Share of letters and digits inside links, scripts and page chrome left out, and the letters counted."""
    body = root.find("body")
    start = body if body is not None else root
    inside = total = 0
    stack = [(start, False)]
    while stack:
        node, in_link = stack.pop()
        tag = node.tag.lower() if isinstance(node.tag, str) else ""
        if node is not start and node.tail:
            count = sum(c.isalnum() for c in node.tail)
            total += count
            inside += count if in_link else 0
        if (
            not tag
            or tag in DROPPED
            or tag in CHROME
            or node.get("role") in CHROME_ROLES
        ):
            continue
        linked = in_link or tag == "a"
        if node.text:
            count = sum(c.isalnum() for c in node.text)
            total += count
            inside += count if linked else 0
        stack.extend((child, linked) for child in reversed(node))
    return (inside / total if total else 0.0), total


def page_features(root: etree._Element) -> Features:
    long_count = long_chars = 0
    for paragraph in UNLINKED_PARAGRAPHS(root):
        length = len(" ".join(paragraph.text_content().split()))
        if length >= LONG_PARAGRAPH:
            long_count += 1
            long_chars += length
    share, visible = visible_link_share(root)
    text = " ".join(body_text(root).split())[:MAX_BODY_CHARS]
    buttons = " ".join(" ".join(b.text_content().split()) for b in BUTTONS(root))
    return Features(
        long_count=long_count,
        long_chars=long_chars,
        paragraph_chars=sum(
            len(" ".join(p.text_content().split())) for p in CHROME_PARAGRAPHS(root)
        ),
        link_share=share,
        visible_chars=visible,
        headline_links=int(HEADLINE_LINKS(root)),
        article_elements=int(ARTICLE_ELEMENTS(root)),
        time_elements=int(TIME_ELEMENTS(root)),
        price_props=bool(PRICE_PROPS(root)),
        prices=len(PRICE.findall(text)),
        cart_buttons=len(CART.findall(buttons)),
        docs_markers=len({m.group(0).lower() for m in DOCS_MARKERS.finditer(text)}),
    )


def body_text(root: etree._Element) -> str:
    """The body's text without scripts, styles and other non-content elements, page chrome included."""
    body = root.find("body")
    start = body if body is not None else root
    parts: list[str] = []
    stack = [start]
    while stack:
        node = stack.pop()
        if isinstance(node, str):
            parts.append(node)
            continue
        tag = node.tag.lower() if isinstance(node.tag, str) else ""
        if tag and tag not in DROPPED:
            parts.append(node.text or "")
            for child in reversed(node):
                if child.tail:
                    stack.append(child.tail)
                stack.append(child)
        parts.append(" ")
    return "".join(parts)
