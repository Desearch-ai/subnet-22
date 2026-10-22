"""The whole page's text, minus navigation, menus, footers, popups, controls, sign-up boxes and consent banners."""

import re

from lxml import etree

from desearch.extraction.text.dom import (
    DROPPED,
    HEADINGS,
    LINK_DENSE_LINKS,
    LINK_DENSE_SHARE,
    MAX_DROPPED_SHARE,
    MAX_TEXT_CHARS,
    NAVIGATION,
    WORD,
    above,
    attached,
    body_copy,
    chars,
    class_words,
    drop_controls,
    hinted,
    inline_text,
    link_share,
    name_words,
    tag_of,
    text_spans,
    unlinked,
)
from desearch.extraction.text.lines import CALL_TO_ACTION, clean, drop_link_runs
from desearch.extraction.text.markdown import LISTS, finish, nested_lists, render

MAX_LISTING_CHARS = 20_000
PAGE_FOOTER_LINKS = 3
MIN_PAGE_CHARS = 300
MIN_GUARDED_PAGE_CHARS = 1000
MAX_FOOTER_TAIL = 0.2
MAX_FORM_SHARE = 0.2
MAX_CONTENT_AREA_SHARE = 0.9
MIN_MENU_LINKS = 8
MENU_LINK_SHARE = 0.6
MIN_FOOTER_TEXT = 300
MIN_CONTACT_TEXT = 60
CONSENT = re.compile(
    r"cookie|consent|gdpr|onetrust|cookiebot|didomi|usercentrics|truste|osano|cmpbox|cmp-banner|cmp-wrapper",
    re.I,
)
MENU_WORDS = frozenset(
    (
        "menu",
        "menus",
        "megamenu",
        "navbar",
        "nav",
        "navigation",
        "drawer",
        "offcanvas",
        "hamburger",
        "social",
    )
)
PAGE_TOP_WORDS = frozenset(("header", "masthead", "topbar"))
FOOTER_WORDS = frozenset(("footer", "prefooter", "colophon"))
POPUP_WORDS = frozenset(
    ("modal", "modals", "popup", "popups", "lightbox", "announcement", "toast")
)
OTHER_PRODUCT_STEMS = ("related", "recommend", "upsell", "crosssell")
OTHER_PRODUCT_PAIRS = frozenset(
    (
        ("also", "like"),
        ("may", "like"),
        ("also", "purchased"),
        ("also", "bought"),
        ("recently", "viewed"),
    )
)
BREADCRUMB_WORDS = frozenset(("breadcrumb", "breadcrumbs", "crumbs"))
CONTACT = re.compile(
    r"[\w.+-]+@[\w-]+\.[\w.]+|\(?\d{3}\)?[ .-]\d{3}[ .-]\d{4}|\bP\.?O\.? box\b|\+\d{1,3}[ \d]{8,}",
    re.I,
)


def is_consent(node: etree._Element) -> bool:
    return bool(tag_of(node)) and bool(
        CONSENT.search(f"{node.get('id', '')} {node.get('class', '')}")
    )


def is_page_footer(node: etree._Element) -> bool:
    links = node.findall(".//a")
    return len(links) >= PAGE_FOOTER_LINKS or (
        bool(links) and link_share(node) >= MAX_DROPPED_SHARE
    )


def is_menu(node: etree._Element, content: set) -> bool:
    tag = tag_of(node)
    if tag == "header" and any(
        tag_of(a) in ("section", "article") for a in node.iterancestors()
    ):
        return False
    names = f"{node.get('id', '')} {node.get('class', '')} {node.get('aria-label', '')}"
    hinted = tag and (
        tag == "header"
        or node.get("role") == "banner"
        or MENU_WORDS.intersection(w.lower() for w in WORD.findall(names))
        or PAGE_TOP_WORDS.intersection(re.split(r"[\s_-]+", names.lower()))
    )
    if not hinted or node in content:
        return False
    return link_share(node) >= MAX_DROPPED_SHARE


def carries_content(node: etree._Element, footer: bool) -> bool:
    """A menu with the page's headline in it, or a footer with real text, which only loses its link lists."""
    if node.find(".//h1") is not None:
        return True
    if not footer:
        return False
    text = unlinked(node)
    return text >= MIN_FOOTER_TEXT or (
        text >= MIN_CONTACT_TEXT and bool(CONTACT.search(node.text_content() or ""))
    )


def drop_link_lists(node: etree._Element) -> None:
    """Drops the link lists, forms and sign-up widgets inside a chrome block that keeps its other text."""
    for block in list(
        node.iter("nav", "div", "ul", "ol", "section", "form", "aside", "p")
    ):
        if block is node or not attached(block, node):
            continue
        links = len(block.findall(".//a"))
        if (
            tag_of(block) == "form"
            or hinted(block)
            or CALL_TO_ACTION.match(inline_text(block))
            or (links >= LINK_DENSE_LINKS and link_share(block) >= LINK_DENSE_SHARE)
            or is_menu_tree(block)
        ):
            block.drop_tree()
    for heading in [h for h in node.iter(*HEADINGS) if attached(h, node)]:
        parent = heading.getparent()
        if (
            parent is not None
            and parent is not node
            and chars(parent) == chars(heading)
        ):
            heading.drop_tree()


def is_menu_tree(node: etree._Element) -> bool:
    """A list of links with lists of links inside it, the shape of a site menu."""
    if tag_of(node) not in LISTS:
        return False
    if not any(
        next(nested_lists(item), None) is not None
        for item in node
        if tag_of(item) == "li"
    ):
        return False
    return (
        len(node.findall(".//a")) >= MIN_MENU_LINKS
        and link_share(node) >= MENU_LINK_SHARE
    )


def other_products(node: etree._Element) -> bool:
    """Related, recommended or recently viewed products, which belong to other pages."""
    words = class_words(node)
    return any(w.startswith(OTHER_PRODUCT_STEMS) for w in words) or any(
        pair in OTHER_PRODUCT_PAIRS for pair in zip(words, words[1:])
    )


def page_chrome(
    tree: etree._Element, page_chars: int, product: bool = False
) -> list[etree._Element]:
    """Navigation, menus, footers, popups, breadcrumbs, banners, small forms and hidden blocks outside the content."""
    content = {
        n
        for area in tree.iter("article", "main")
        if chars(area) < MAX_CONTENT_AREA_SHARE * page_chars
        for n in area.iter()
        if n is not area
    }
    spans = text_spans(tree)
    end = spans[tree][1]
    navs = set(NAVIGATION(tree))
    tabled = {n for table in tree.iter("table") for n in table.iter() if n is not table}
    chrome = []
    for node in tree.iter():
        tag = tag_of(node)
        if not tag or node is tree or node in tabled:
            continue
        words = name_words(node)
        footer = tag == "footer" or (
            bool(FOOTER_WORDS & words) and end - spans[node][1] < MAX_FOOTER_TAIL * end
        )
        outside = node not in content
        if node in navs and not content_nav(node):
            chrome.append(node)
        elif footer and is_page_footer(node) and outside:
            chrome.append(node)
        elif is_consent(node) or is_menu(node, content):
            chrome.append(node)
        elif outside and (POPUP_WORDS & words or BREADCRUMB_WORDS & words):
            chrome.append(node)
        elif tag == "form" and outside and chars(node) <= MAX_FORM_SHARE * page_chars:
            chrome.append(node)
        elif tag == "marquee" or (
            product and other_products(node) and link_share(node) >= MAX_DROPPED_SHARE
        ):
            chrome.append(node)
    return chrome


def content_nav(nav: etree._Element) -> bool:
    """A nav inside the main area that holds most of its text, like a services directory."""
    area = next(
        (a for a in nav.iterancestors() if tag_of(a) in ("main", "article")), None
    )
    return area is not None and chars(nav) > MAX_DROPPED_SHARE * chars(area)


def full_text(
    root: etree._Element, title: str, cap: int = MAX_TEXT_CHARS, product: bool = False
) -> str:
    """The whole page in reading order, minus navigation, menus, footers, popups, small forms and consent banners."""
    tree = body_copy(root, DROPPED)
    drop_controls(tree)
    page_chars = left = chars(tree)
    headline = next(iter(tree.iter("h1")), None)
    for node in page_chrome(tree, page_chars, product):
        if not attached(node, tree):
            continue
        size = chars(node)
        # A block holding most of the page goes only if it is all links above the headline, like a mega-menu.
        if size > page_chars * MAX_DROPPED_SHARE and (
            link_share(node) < LINK_DENSE_SHARE or not above(node, headline)
        ):
            continue
        # Never leave a real page nearly empty: the block is then its content.
        if page_chars >= MIN_GUARDED_PAGE_CHARS and left - size < MIN_PAGE_CHARS:
            continue
        footer = tag_of(node) == "footer" or bool(FOOTER_WORDS & name_words(node))
        if not is_consent(node) and carries_content(node, footer):
            drop_link_lists(node)
            continue
        node.drop_tree()
        left -= size
    return finish(
        title, drop_link_runs(clean(render(tree, cards=True), cards=True)), cap
    )
