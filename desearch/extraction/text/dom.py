"""Element helpers shared by the text modes: an element's text and links, its names, hidden text, and the cleaned body copy."""

import re
from copy import deepcopy

from lxml import etree

MAX_TEXT_CHARS = 200_000
MAX_DROPPED_SHARE = 0.5
MAX_CONTROL_WORDS = 8
MAX_LABEL_WORDS = 12
LINK_DENSE_SHARE = 0.7
LINK_DENSE_LINKS = 3
MIN_TITLE_SHARE = 0.8

HEADINGS = {"h1": 1, "h2": 2, "h3": 3, "h4": 4, "h5": 5, "h6": 6}
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
    "input",
    "textarea",
    "link",
    "meta",
    "video",
    "audio",
    "title",
)
MAIN_DROPPED = (*DROPPED, "button")
CODE = ("script", "style", "noscript", "template")
BLOCKS = frozenset(
    (
        "address",
        "article",
        "aside",
        "blockquote",
        "button",
        "caption",
        "center",
        "dd",
        "details",
        "div",
        "dl",
        "dt",
        "fieldset",
        "figcaption",
        "figure",
        "footer",
        "form",
        "header",
        "hr",
        "label",
        "legend",
        "li",
        "main",
        "nav",
        "ol",
        "p",
        "pre",
        "section",
        "summary",
        "table",
        "td",
        "th",
        "tr",
        "ul",
    )
)
SPACED = frozenset(("a", "time", "img", "small", "cite", "abbr", "sup", "sub", "data"))
BLOCKY = re.compile(r"(^|\s)(block|flex|grid|inline-block|inline-flex)(\s|$)")
CLOSING_PUNCTUATION = frozenset(".,;:!?)]’”»%")
CONTROLS = frozenset(("button", "label", "summary"))

SPACES = re.compile(r"[ \t\r\n\f\v ​‌‍⁠﻿]+")
SPACE_BEFORE_PUNCT = re.compile(r" ([,;:!?)\]”»]|\.(?!\w))")
SPACE_BEFORE_APOSTROPHE = re.compile(r" ’(?=s\b|\W|$)")
SPACE_AFTER_OPEN = re.compile(r"([(\[“‘«]) ")
REPEATED_COMMAS = re.compile(r"(, )+")
FOOTNOTE = re.compile(r"^(\[\s*\w{1,4}\s*\]|\d{1,3}|\*)$")
HINT_ATTRIBUTES = (
    "class",
    "id",
    "data-testid",
    "data-test",
    "data-qa",
    "data-cy",
    "data-component",
)
WORD = re.compile(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|\d+")
NEGATED = re.compile(r"^(no|without)[-_]", re.I)
WP_STATUS = re.compile(r"^status-(publish|private|future|draft)$")
WP_TYPE = re.compile(r"^type-[\w-]+$")
HINT_WORDS = frozenset(
    (
        "share",
        "sharing",
        "social",
        "comment",
        "comments",
        "cmt",
        "disqus",
        "subscribe",
        "subscription",
        "subscriber",
        "donate",
        "donation",
        "donations",
        "tags",
        "topics",
        "categories",
        "crumb",
        "crumbs",
        "trending",
        "popular",
        "tooltip",
        "popover",
        "jwplayer",
        "pagination",
        "pager",
        "search",
    )
)
HINT_STEMS = (
    "related",
    "recirc",
    "recommend",
    "newsletter",
    "promo",
    "sponsor",
    "advert",
    "breadcrumb",
    "outbrain",
    "taboola",
    "pullquote",
    "relpost",
)
HINT_PAIRS = frozenset(
    (
        ("more", "stories"),
        ("read", "next"),
        ("read", "more"),
        ("also", "like"),
        ("may", "like"),
        ("more", "topic"),
        ("more", "topics"),
        ("preferred", "source"),
        ("audio", "player"),
        ("video", "player"),
        ("media", "player"),
        ("post", "navigation"),
        ("post", "nav"),
        ("nav", "links"),
        ("pull", "quote"),
        ("print", "only"),
    )
)
AD_WORDS = frozenset(("ad", "ads", "adslot", "adunit"))
HIDDEN_CLASS = re.compile(
    r"^(.*[-_])?(sr-only|visually-hidden|visuallyhidden|screen-reader-text|screen-reader-only|element-invisible)$"
)
ICON_FONTS = ("material-icons", "material-symbols")
LIGATURE = re.compile(r"^[a-z][a-z_]*$")
LAST_WORD = re.compile(r"(\w{2,}|[,;:])$")
FIRST_WORD = re.compile(r"^\w{2,}")
WORDS = re.compile(r"\w+")
VALUE = re.compile(r"[\d$€£¥%]")
BYLINED = etree.XPath(".//time | .//a[@rel='author' or contains(@href, '/author')]")
BYLINE_NAME = re.compile(r"\bby [A-Z][\w'.-]+ [A-Z]")
A11Y_MARKED = etree.XPath("//*[contains(@id, 'a11y') or contains(@class, 'a11y')]")
A11Y_NAME = re.compile(r"(^|[\s_-])a11y([\s_-]|$)", re.I)
MAX_SIGNUP_CHARS = 400
MAX_SIGNUP_EXTRA_CHARS = 200
NAVIGATION = etree.XPath(
    "//nav | //*[@role='navigation' or @role='contentinfo' or @role='search']"
)
DIALOGS = etree.XPath(
    "//dialog | //*[@role='dialog' or @role='alertdialog' or @aria-modal='true']"
)
CONTROL_LABELS = etree.XPath("//button | //*[@aria-live or @role='status']")
CLOUDFLARE_EMAILS = etree.XPath("//*[@data-cfemail]")
MARKED = etree.XPath("descendant-or-self::*[@class or @role or @data-component]")
OUTER_LINKS = etree.XPath("descendant-or-self::a[not(ancestor::a)]")


def tag_of(node: etree._Element) -> str:
    return node.tag.lower() if isinstance(node.tag, str) else ""


def tidy(text: str) -> str:
    line = SPACE_BEFORE_APOSTROPHE.sub(
        "’", SPACE_BEFORE_PUNCT.sub(r"\1", SPACES.sub(" ", text).strip())
    )
    return SPACE_AFTER_OPEN.sub(r"\1", line).strip()


def chars(node: etree._Element) -> int:
    return len(SPACES.sub(" ", node.text_content()).strip())


def alnum(text: str | None) -> int:
    return sum(map(str.isalnum, text or ""))


def link_share(node: etree._Element) -> float:
    total = alnum(node.text_content())
    if not total:
        return 0.0
    linked = sum(alnum(a.text_content()) for a in OUTER_LINKS(node))
    return linked / total


def inline_text(
    node: etree._Element, list_sep: str = ", ", skip: tuple[str, ...] = ()
) -> str:
    """The node's text on one line: list items joined by `list_sep`, blocks by spaces, `skip` subtrees left out."""
    parts: list[str] = []

    def walk(n: etree._Element) -> None:
        tag = tag_of(n)
        if tag == "br" or (spaced(n) and not opens_with_punctuation(n)):
            parts.append(" ")
        if n.text:
            parts.append(n.text)
        for child in n:
            child_tag = tag_of(child)
            if child_tag and child_tag not in skip:
                walk(child)
                parts.append(
                    list_sep
                    if child_tag == "li"
                    else " "
                    if child_tag in BLOCKS
                    else ""
                )
            if child.tail:
                parts.append(child.tail)
        if spaced(n):
            parts.append(" ")

    walk(node)
    joined = REPEATED_COMMAS.sub(
        ", ", SPACES.sub(" ", "".join(parts)).strip().strip(" ,")
    )
    return tidy(joined)


def spaced(node: etree._Element) -> bool:
    """Inline elements a browser lays out apart, so their words never run together."""
    tag = tag_of(node)
    return tag in SPACED or (
        tag == "span" and bool(BLOCKY.search(node.get("class") or ""))
    )


def opens_with_punctuation(node: etree._Element) -> bool:
    """Text like ".Selector" that continues the word before it, so padding would split it."""
    return (node.text_content() or "")[:1] in CLOSING_PUNCTUATION


def link_only(node: etree._Element) -> bool:
    total = alnum(node.text_content())
    return total > 0 and link_share(node) >= 1.0


def text_start(node: etree._Element, size: int = 120) -> str:
    """The first words of an element's text, without reading all of it."""
    parts, total = [], 0
    for piece in node.itertext():
        parts.append(piece)
        total += len(piece)
        if total >= size:
            break
    return " ".join("".join(parts).split()).lstrip("-–•* ")


def class_names(node: etree._Element) -> str:
    return " ".join(node.get(name) or "" for name in HINT_ATTRIBUTES)


def class_words(node: etree._Element) -> list[str]:
    return [word.lower() for word in WORD.findall(class_names(node))]


def hinted(node: etree._Element) -> bool:
    """Share, related, ad, comment and similar widgets, by the words of their class, id and test names."""
    tokens = (node.get("class") or "").split()
    # WordPress names a post by its tags and topics ("tag-popular"): the post itself is no widget.
    if any(WP_STATUS.match(t) for t in tokens) and any(
        WP_TYPE.match(t) for t in tokens
    ):
        return False
    # A "no-share" or "no-comments" class names a widget the element lacks.
    names = " ".join(n for n in class_names(node).split() if not NEGATED.match(n))
    words = [word.lower() for word in WORD.findall(names)]
    first_words = {
        found[0].lower() for found in map(WORD.findall, names.split()) if found
    }
    return (
        any(w in HINT_WORDS or w.startswith(HINT_STEMS) for w in words)
        or any(pair in HINT_PAIRS for pair in zip(words, words[1:]))
        or bool(AD_WORDS.intersection(first_words))
    )


def in_running_text(node: etree._Element) -> bool:
    """An inline element inside prose or code, which a drop would cut out of its sentence."""
    if any(tag_of(a) in ("pre", "code") for a in node.iterancestors()):
        return True
    if tag_of(node) in BLOCKS:
        return False
    parent = node.getparent()
    if parent is None:
        return False
    return bool((parent.text or "").strip()) or any(
        (child.tail or "").strip() for child in parent
    )


def hidden_label(node: etree._Element) -> bool:
    """Screen-reader text and tooltips: labels in an article, but the only copy of ratings and details on some pages."""
    return node.get("role") == "tooltip" or any(
        HIDDEN_CLASS.match(name) for name in (node.get("class") or "").lower().split()
    )


def in_control(node: etree._Element) -> bool:
    return any(
        tag_of(a) in CONTROLS or a.get("role") == "button" for a in node.iterancestors()
    )


def icon_text(node: etree._Element) -> bool:
    """An icon font's ligature, like "chevron_right", which a reader sees as a glyph."""
    names = (node.get("class") or "").lower().split()
    icon = node.get("data-component") == "icon" or any(
        n.startswith(ICON_FONTS) for n in names
    )
    return icon and bool(LIGATURE.match((node.text_content() or "").strip()))


def space_items(tree: etree._Element) -> None:
    """Separates inline elements that sit side by side in a block with no text of its own, as layout lays them apart."""
    code = {n for block in tree.iter("pre", "code") for n in block.iter()}
    for parent in tree.iter():
        tag = tag_of(parent)
        if (
            (tag not in BLOCKS and tag not in HEADINGS)
            or parent in code
            or (parent.text or "").strip()
        ):
            continue
        kids = [k for k in parent if tag_of(k)]
        if len(kids) < 2:
            continue
        if any((k.tail or "").strip() for k in kids):
            continue
        previous_text = ""
        for i, kid in enumerate(kids):
            text = kid.text_content() or ""
            if not text.strip():
                continue
            if (
                previous_text
                and not previous_text[-1].isspace()
                and not text[0].isspace()
                and LAST_WORD.search(previous_text)
                and FIRST_WORD.match(text)
                and not kids[i - 1].tail
            ):
                kids[i - 1].tail = " "
            previous_text = text
    for node in tree.iter():
        parent = node.getparent()
        if (
            parent is None
            or tag_of(node) in ("", "wbr")
            or not node.tail
            or node in code
        ):
            continue
        if len(node) or (node.text or "").strip():
            continue
        before = node.getprevious()
        preceding = before.tail if before is not None else parent.text
        if preceding and LAST_WORD.search(preceding) and FIRST_WORD.match(node.tail):
            node.tail = " " + node.tail


def cloudflare_email(encoded: str) -> str:
    """The address Cloudflare hides as hex, each byte XORed with the first."""
    try:
        data = bytes.fromhex(encoded)
        return bytes(b ^ data[0] for b in data[1:]).decode("utf-8")
    except (ValueError, IndexError, UnicodeDecodeError):
        return ""


def drop_signup_boxes(tree: etree._Element) -> None:
    """The small box around a lone email field, a newsletter or sign-up prompt: its form, heading and a line or two."""
    for field in tree.xpath(".//input[translate(@type, 'EMAIL', 'email')='email']"):
        if not attached(field, tree):
            continue
        form = next(
            (a for a in field.iterancestors() if tag_of(a) == "form"), field.getparent()
        )
        limit = min(MAX_SIGNUP_CHARS, chars(form) + MAX_SIGNUP_EXTRA_CHARS)
        box = None
        for node in field.iterancestors():
            if node is tree or node.find(".//textarea") is not None:
                break
            if (
                len(
                    node.xpath(
                        ".//input[not(@type) or @type='text' or @type='email' or @type='tel']"
                    )
                )
                > 2
            ):
                break
            if (
                chars(node) > limit
                or BYLINED(node)
                or BYLINE_NAME.search(inline_text(node))
            ):
                break
            box = node
        if box is not None:
            box.drop_tree()


class Valued:
    """A tooltip, or screen-reader text beside a value or long enough to be content itself; parents are read once."""

    def __init__(self) -> None:
        self.parents: dict[etree._Element, bool] = {}

    def __call__(self, node: etree._Element) -> bool:
        if (
            node.get("role") == "tooltip"
            or len(node.text_content().split()) > MAX_LABEL_WORDS
        ):
            return True
        parent = node.getparent()
        parent = parent if parent is not None else node
        if parent not in self.parents:
            self.parents[parent] = bool(VALUE.search(parent.text_content()))
        return self.parents[parent]


def body_copy(
    root: etree._Element, dropped: tuple[str, ...], labels: bool = True
) -> etree._Element:
    """The body without `dropped` elements, sign-up boxes, dialogs, icon and a11y text, footnote marks, and hidden labels unless `labels`."""
    body = root.find("body")
    tree = deepcopy(body if body is not None else root)
    etree.strip_elements(tree, *CODE, with_tail=False)
    drop_signup_boxes(tree)
    etree.strip_elements(tree, *dropped, with_tail=False)
    valued = Valued()
    hidden = [
        n
        for n in MARKED(tree)
        if icon_text(n)
        or (hidden_label(n) and (not labels or in_control(n) or not valued(n)))
    ]
    hidden += [
        n
        for n in A11Y_MARKED(tree)
        if A11Y_NAME.search(f"{n.get('id', '')} {n.get('class', '')}") and not valued(n)
    ]
    for node in DIALOGS(tree) + hidden:
        if node.getparent() is not None:
            node.drop_tree()
    for sup in list(tree.iter("sup")):
        if FOOTNOTE.match(inline_text(sup)) and any(
            (a.get("href") or "").startswith("#") for a in sup.iter("a")
        ):
            sup.drop_tree()
    for node in CLOUDFLARE_EMAILS(tree):
        node.text, node[:] = cloudflare_email(node.get("data-cfemail") or ""), []
    space_items(tree)
    return tree


def attached(node: etree._Element, tree: etree._Element) -> bool:
    return node is tree or any(a is tree for a in node.iterancestors())


def unlinked(node: etree._Element) -> int:
    """Letters and digits outside links."""
    return alnum(node.text_content()) - sum(
        alnum(a.text_content()) for a in OUTER_LINKS(node)
    )


def text_spans(tree: etree._Element) -> dict[etree._Element, tuple[int, int]]:
    """Where each element's text starts and ends in the page's text."""
    spans: dict[etree._Element, tuple[int, int]] = {}
    position = 0

    def walk(node: etree._Element) -> None:
        nonlocal position
        start = position
        position += len(node.text or "") if tag_of(node) else 0
        for child in node:
            walk(child)
            position += len(child.tail or "")
        spans[node] = (start, position)

    walk(tree)
    return spans


def name_words(node: etree._Element) -> set[str]:
    return {
        w.lower() for w in WORD.findall(f"{node.get('id', '')} {node.get('class', '')}")
    }


def above(node: etree._Element, headline: etree._Element | None) -> bool:
    """Comes before the page's h1 in reading order, or the page has none."""
    if headline is None:
        return True
    if headline in node.iter():
        return False
    for element in node.getroottree().iter():
        if element is node:
            return True
        if element is headline:
            return False
    return False


def drop_controls(tree: etree._Element) -> None:
    """Short buttons and status announcements; a button asking a question, like an FAQ's, stays."""
    for node in CONTROL_LABELS(tree):
        if node is tree or not attached(node, tree):
            continue
        text = " ".join(node.text_content().split())
        if len(text.split()) > MAX_CONTROL_WORDS:
            continue
        if tag_of(node) == "button" and text.endswith("?"):
            continue
        node.drop_tree()


def same_title(heading: str, title: str) -> bool:
    words, titled = (
        WORDS.findall(heading.casefold()),
        set(WORDS.findall(title.casefold())),
    )
    return bool(words) and sum(w in titled for w in words) >= MIN_TITLE_SHARE * len(
        words
    )


def headline(tree: etree._Element, title: str) -> etree._Element | None:
    """The h1 that names the page, else an h2 that does, else the first h1."""
    headings = [h for h in tree.iter("h1") if inline_text(h)]
    named = [
        h for h in (*headings, *tree.iter("h2")) if same_title(inline_text(h), title)
    ]
    return (named or headings or [None])[0]
