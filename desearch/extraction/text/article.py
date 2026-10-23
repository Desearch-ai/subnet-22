"""The main content of an article or document: its scope, headline and byline, without widgets and link boxes."""

import re
from urllib.parse import urlsplit

from lxml import etree

from desearch.extraction.text.dom import (
    DROPPED,
    HEADINGS,
    LINK_DENSE_LINKS,
    LINK_DENSE_SHARE,
    MAX_DROPPED_SHARE,
    MAX_TEXT_CHARS,
    NAVIGATION,
    body_copy,
    chars,
    class_words,
    drop_controls,
    headline,
    hinted,
    in_running_text,
    inline_text,
    link_share,
    same_title,
    tag_of,
    text_start,
)
from desearch.extraction.text.lines import (
    APPEAL,
    CALL_TO_ACTION,
    clean,
    drop_author_bio,
    drop_orphans,
)
from desearch.extraction.text.markdown import Line, finish, render

MAX_SCOPE_DROP_SHARE = 0.3
MIN_SCOPE_SHARE = 0.6
MIN_CHILD_SHARE = 0.75
MIN_DOMINANT_SHARE = 0.8
MIN_DOMINANT_PROSE = 1000
MAX_DOMINANT_TEXT_SHARE = 2
MIN_DOMINANT_PARAGRAPHS = 2
MIN_BODY_SHARE = 0.5
MIN_CONTAINER_SHARE = 0.5
MIN_CONTAINER_CHARS = 200
MAX_LINK_BOX_CHARS = 300
LONG_PARAGRAPH_CHARS = 120
MAX_BYLINE_LINKS = 4
MAX_BYLINE_LINK_WORDS = 5
MAX_HEADER_CHARS = 1500
# Ranked: names only a post's body has, then a story's, then a page's content.
CONTENT_CONTAINERS = (
    re.compile(
        r"^(post|entry|hentry)$|(entry|article|post|page|text|art)[-_]?content|article[-_]?(body|text)|articlebody"
        r"|post[-_]?(body|text|entry)|postcontent|body[-_]text|article__(content|body)",
        re.I,
    ),
    re.compile(
        r"^(story|postarea|text)$|story[-_]?(body|content)|(theme|blog|section|single)[-_]content|single[-_]post"
        r"|fulltext|field[-_]body",
        re.I,
    ),
    re.compile(
        r"^content$|content[-_](main|body)|contentbody|main[-_]content|page[-_]content",
        re.I,
    ),
)
SCOPE_HINT = re.compile(
    r"(^|[\s_-])(share|sharing|social|related|newsletter|subscribe|promo|advert|ad-slot|ads|comments?|recirc"
    r"|more-stories|read-next|outbrain|taboola|breadcrumbs?)([\s_-]|$)",
    re.I,
)
HEADER_WIDGETS = frozenset(("widget", "widgets"))
AUTHOR_PATH = re.compile(r"/authors?/")
BYLINE_TEXT = re.compile(r"^[Bb][Yy] [A-Z]")
PREFERRED_SOURCE = "google.com/preferences/source"
DOCUMENT_LINK = re.compile(r"\.(pdf|docx?|xlsx?|pptx?|csv|zip|epub)([?#]|$)", re.I)
FILE_TEXT = re.compile(r"\b(pdf|docx?|xlsx?|pptx?|csv|zip|epub)\b", re.I)
URL_TEXT = re.compile(r"^(https?://|www\.)", re.I)
MAX_APPEAL_CHARS = 1500
ARTICLE_BODY = etree.XPath("//*[@itemprop='articleBody']")
MAINS = etree.XPath("//main | //*[@role='main']")
CHROME_PARAGRAPHS = etree.XPath(
    "//p[not(ancestor::nav or ancestor::header or ancestor::footer or ancestor::aside)]"
)


def scope(tree: etree._Element, article: bool = False) -> etree._Element | None:
    """The element holding the main content: articleBody, the dominant <article> or <main>, the deepest element holding most of the prose, else a container named like content."""
    sizes = {p: chars(p) for p in CHROME_PARAGRAPHS(tree)}
    prose = sum(sizes.values()) or 1
    under, _ = prose_under(sizes)

    def held(node: etree._Element) -> int:
        return under.get(node, 0)

    body = ARTICLE_BODY(tree)
    if body and held(body[0]) >= MIN_BODY_SHARE * prose:
        parent = body[0].getparent()
        return parent if parent is not None and tag_of(parent) == "article" else body[0]
    articles = sorted(tree.iter("article"), key=held, reverse=True)
    if articles and held(articles[0]) >= MIN_SCOPE_SHARE * prose:
        return articles[0]
    mains = MAINS(tree)
    if mains and held(mains[0]) >= MIN_SCOPE_SHARE * prose:
        return mains[0]
    if prose < MIN_DOMINANT_PROSE:
        # Text in divs rather than paragraphs: the <article> or <main> holding most of the page's text.
        whole = chars(tree) or 1
        for node in (*articles[:1], *mains[:1]):
            if chars(node) >= MIN_SCOPE_SHARE * whole:
                return node
    kept = outside_widgets(tree, sizes)
    found = dominant(tree, kept)
    return found if found is not None else named_container(tree, kept)


def prose_under(
    sizes: dict[etree._Element, int],
) -> tuple[dict[etree._Element, int], dict[etree._Element, int]]:
    """The paragraph text under each element, and how many paragraphs."""
    under: dict[etree._Element, int] = {}
    count: dict[etree._Element, int] = {}
    for p, size in sizes.items():
        for node in (p, *p.iterancestors()):
            under[node] = under.get(node, 0) + size
            count[node] = count.get(node, 0) + 1
    return under, count


def outside_widgets(
    tree: etree._Element, sizes: dict[etree._Element, int]
) -> dict[etree._Element, int]:
    """The paragraphs outside comments, related links and other widgets, with their sizes."""
    widget: dict[etree._Element, bool] = {}

    def in_widget(p: etree._Element) -> bool:
        return any(
            widget.setdefault(a, hinted(a)) for a in p.iterancestors() if a is not tree
        )

    return {p: size for p, size in sizes.items() if not in_widget(p)}


def named_container(
    tree: etree._Element, kept: dict[etree._Element, int]
) -> etree._Element | None:
    """The first element named like a post's body, then a page's content, holding most of the prose (or text, without paragraphs) outside widgets."""
    prose = sum(kept.values())
    if prose >= MIN_DOMINANT_PROSE:
        under, _ = prose_under(kept)

        def held(node: etree._Element) -> int:
            return under.get(node, 0)

        total = prose
    else:
        widgets: dict[etree._Element, bool] = {}
        total = content_chars(tree, widgets)

        def held(node: etree._Element) -> int:
            return content_chars(node, widgets)

        if total < MIN_CONTAINER_CHARS:
            return None
    for names in CONTENT_CONTAINERS:
        for node in tree.iter("article", "div", "main", "section"):
            if (
                any(
                    names.search(name)
                    for name in f"{node.get('id', '')} {node.get('class', '')}".split()
                )
                and held(node) >= MIN_CONTAINER_SHARE * total
            ):
                return node
    return None


def content_chars(node: etree._Element, widgets: dict[etree._Element, bool]) -> int:
    """An element's text outside the widgets inside it; `widgets` caches which elements are widgets."""
    outer = []
    for n in node.iterdescendants():
        if (
            tag_of(n)
            and widgets.setdefault(n, hinted(n))
            and not in_widget(n, node, widgets)
        ):
            outer.append(n)
    return chars(node) - sum(chars(w) for w in outer)


def in_widget(
    n: etree._Element, top: etree._Element, widgets: dict[etree._Element, bool]
) -> bool:
    for a in n.iterancestors():
        if a is top:
            return False
        if widgets.get(a):
            return True
    return False


def dominant(
    tree: etree._Element, kept: dict[etree._Element, int]
) -> etree._Element | None:
    """The deepest element holding most of the prose outside comments, related links and other widgets."""
    prose = sum(kept.values())
    if prose < MIN_DOMINANT_PROSE:
        return None
    under, count = prose_under(kept)
    node = tree
    while True:
        best = max(
            (c for c in node if tag_of(c)), key=lambda c: under.get(c, 0), default=None
        )
        if (
            best is None
            or under.get(best, 0) < MIN_DOMINANT_SHARE * prose
            or under.get(best, 0) < MIN_CHILD_SHARE * under.get(node, 0)
        ):
            break
        node = best
    if node is tree or count.get(node, 0) < MIN_DOMINANT_PARAGRAPHS:
        return None
    return node if chars(node) <= MAX_DOMINANT_TEXT_SHARE * under[node] else None


def bylined(node: etree._Element) -> bool:
    """Holds one date, an author link or byline markup, and few, short links."""
    times, links = len(node.findall(".//time")), node.findall(".//a")
    if times > 1 or len(links) > MAX_BYLINE_LINKS:
        return False
    if any(len(inline_text(a).split()) > MAX_BYLINE_LINK_WORDS for a in links):
        return False
    if times or BYLINE_TEXT.match(inline_text(node)):
        return True
    for n in node.iter():
        if not tag_of(n):
            continue
        if n.get("itemprop") in ("author", "datePublished", "dateModified"):
            return True
        if "author" in (n.get("rel") or "") or AUTHOR_PATH.search(n.get("href") or ""):
            return True
        if {"byline", "author", "dateline"}.intersection(class_words(n)):
            return True
    return False


def byline_guard(blocks: list[etree._Element]):
    """Tells whether a block is the byline or date before the body, which drops must keep."""
    order = {n: i for i, n in enumerate(n for block in blocks for n in block.iter())}
    first = next(
        (
            order[p]
            for p in blocks[-1].iter("p")
            if chars(p) >= LONG_PARAGRAPH_CHARS and link_share(p) < MAX_DROPPED_SHARE
        ),
        len(order),
    )

    def guarded(node: etree._Element) -> bool:
        if order.get(node, len(order)) >= first:
            return False
        last = max((order.get(n, 0) for n in node.iter()), default=0)
        return last < first and chars(node) < MAX_LINK_BOX_CHARS and bylined(node)

    return guarded


def header_blocks(
    main: etree._Element, heading: etree._Element | None, title: str
) -> list[etree._Element]:
    """The blocks from the headline to the scope when the headline sits beside it: dek, byline, date and lead image."""
    if heading is None:
        return []
    # A page's only h1 is its headline even when the title is worded for search.
    if (
        not same_title(inline_text(heading), title)
        and sum(1 for _ in heading.getroottree().getroot().iter("h1")) != 1
    ):
        return []
    above = set(main.iterancestors())
    common = next((a for a in heading.iterancestors() if a in above), None)
    if common is None:
        return []
    top = next(
        n for n in (heading, *heading.iterancestors()) if n.getparent() is common
    )
    bottom = next(n for n in (main, *main.iterancestors()) if n.getparent() is common)
    children = list(common)
    start, stop = children.index(top), children.index(bottom)
    blocks = children[start:stop]
    if start >= stop or sum(chars(b) for b in blocks) > MAX_HEADER_CHARS:
        return []
    return blocks


def site_of(url: str) -> str:
    try:
        return (urlsplit(url).hostname or "").removeprefix("www.")
    except ValueError:
        return ""


def stays_on_site(node: etree._Element, site: str) -> bool:
    """Every link stays on the page's site, as related stories and tags do and product links do not."""
    for a in node.iter("a"):
        href = a.get("href") or ""
        host = site_of(href)
        if PREFERRED_SOURCE in href:
            continue
        if (
            host
            and host != site
            and not host.endswith("." + site)
            and not site.endswith("." + host)
        ):
            return False
    return True


def outside_body(main: etree._Element):
    """Tells whether a block comes before the article's first body paragraph or after its last."""
    order = {n: i for i, n in enumerate(main.iter())}
    body = [
        order[p]
        for p in main.iter("p")
        if chars(p) >= LONG_PARAGRAPH_CHARS and link_share(p) < MAX_DROPPED_SHARE
    ]
    if not body:
        return lambda node: False
    first, last = min(body), max(body)
    return lambda node: not first <= order.get(node, -1) <= last


def prune_document(block: etree._Element, total: int) -> None:
    """Drops share, related and similar widgets and link lists from a docs page's main content."""
    widgets = [n for n in block.iter("aside", "form", "footer")] + [
        n
        for n in block.iter()
        if tag_of(n) and SCOPE_HINT.search(f"{n.get('class', '')} {n.get('id', '')}")
    ]
    for node in widgets:
        if node.getparent() is not None and chars(node) < MAX_SCOPE_DROP_SHARE * total:
            node.drop_tree()
    for node in list(block.iter("div", "ul", "ol", "section")):
        if (
            node.getparent() is not None
            and len(node.findall(".//a")) >= LINK_DENSE_LINKS
            and link_share(node) >= LINK_DENSE_SHARE
            and chars(node) < MAX_SCOPE_DROP_SHARE * total
        ):
            node.drop_tree()


def prune(
    block: etree._Element,
    total: int,
    guarded,
    edge,
    site: str,
    header: bool = False,
) -> None:
    """Drops widgets, link lists, and the small link boxes and calls to action before and after the body, from one block of an article."""
    widgets = [n for n in block.iter("aside", "form", "footer")] + [
        n
        for n in block.iter()
        if tag_of(n)
        and tag_of(n) not in HEADINGS
        and (hinted(n) or (header and HEADER_WIDGETS.intersection(class_words(n))))
        and not in_running_text(n)
    ]
    for node in widgets:
        if (
            node.getparent() is not None
            and not guarded(node)
            and chars(node) < MAX_SCOPE_DROP_SHARE * total
        ):
            node.drop_tree()
    for node in list(block.iter("div", "ul", "ol", "section")):
        if (
            node.getparent() is not None
            and len(node.findall(".//a")) >= LINK_DENSE_LINKS
            and link_share(node) >= LINK_DENSE_SHARE
            and chars(node) < MAX_SCOPE_DROP_SHARE * total
            and not any(guarded(n) for n in node.iter())
        ):
            node.drop_tree()
    for node in list(block.iter("div", "ul", "ol", "section", "p", "figure")):
        if (
            node.getparent() is not None
            and node.find(".//a") is not None
            and edge(node)
            and chars(node) < MAX_LINK_BOX_CHARS
            and link_share(node) >= LINK_DENSE_SHARE
            and not any(guarded(n) for n in node.iter())
            and stays_on_site(node, site)
            and not any(tag_of(a) == "table" for a in node.iterancestors())
            and not any(
                URL_TEXT.match(inline_text(a))
                or DOCUMENT_LINK.search(a.get("href") or "")
                or FILE_TEXT.search(inline_text(a))
                for a in node.iter("a")
            )
        ):
            node.drop_tree()
    for node in list(
        block.iter("div", "section", "aside", "p", "ul", "li", "h2", "h3", "h4")
    ):
        if (
            node.getparent() is not None
            and APPEAL.match(text_start(node))
            and chars(node) < MAX_APPEAL_CHARS
            and not any(guarded(n) for n in node.iter())
        ):
            node.drop_tree()
    for node in list(block.iter("div", "section", "aside", "p", "ul", "figure")):
        if (
            node.getparent() is not None
            and edge(node)
            and chars(node) < MAX_LINK_BOX_CHARS
            and CALL_TO_ACTION.match(inline_text(node))
            and not any(guarded(n) for n in node.iter())
        ):
            node.drop_tree()


def main_text(
    root: etree._Element,
    title: str,
    url: str,
    article: bool,
    cap: int = MAX_TEXT_CHARS,
    author: str = "",
) -> str:
    """The article or document itself, or "" when no element holds most of the prose."""
    tree = body_copy(root, DROPPED, labels=not article)
    drop_controls(tree)
    for node in NAVIGATION(tree):
        if node.getparent() is not None:
            node.drop_tree()
    main = scope(tree, article)
    if main is None:
        return ""
    heading = headline(tree, title)
    heading_text = inline_text(heading) if heading is not None else ""
    blocks = [*header_blocks(main, heading, title), main]
    total = chars(main)
    if article:
        guarded, edge, site = byline_guard(blocks), outside_body(main), site_of(url)
        for block in blocks:
            prune(block, total, guarded, edge, site, header=block is not main)
        lines = drop_author_bio(
            drop_orphans(clean([line for block in blocks for line in render(block)])),
            author,
        )
    else:
        for block in blocks:
            prune_document(block, total)
        lines = clean([line for block in blocks for line in render(block)])
    if (
        heading_text
        and not any(line.text.startswith("# ") for line in lines)
        and heading_text not in (line.text.lstrip("# ") for line in lines)
    ):
        lines = [Line(f"# {heading_text}"), *lines]
    return finish(title, lines, cap)
