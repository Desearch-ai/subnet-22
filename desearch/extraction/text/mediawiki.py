"""Wikipedia and other MediaWiki pages: lead, infobox, sections and categories."""

from copy import deepcopy

from lxml import etree

from desearch.extraction.text.dom import (
    HEADINGS,
    MAIN_DROPPED,
    MAX_TEXT_CHARS,
    inline_text,
    tag_of,
)
from desearch.extraction.text.lines import clean
from desearch.extraction.text.markdown import Line, finish, render

MW_CONTENT = etree.XPath(
    "//*[@id='mw-content-text']//*[contains(concat(' ', normalize-space(@class), ' '), ' mw-parser-output ')]"
)
MW_CATEGORIES = etree.XPath("//*[@id='mw-normal-catlinks']//li//a")
MW_SHORT = etree.XPath(
    "//*[contains(concat(' ', normalize-space(@class), ' '), ' shortdescription ')]"
)
MW_INFOBOXES = etree.XPath(
    ".//table[contains(concat(' ', normalize-space(@class), ' '), ' infobox ')]"
)
# Footnote marks and inline cleanup tags like "[citation needed]".
MW_MARKS = etree.XPath(
    ".//*[contains(concat(' ', normalize-space(@class), ' '), ' reference ')"
    " or contains(concat(' ', normalize-space(@class), ' '), ' Inline-Template ')]"
)
MW_CAPTIONS = etree.XPath(".//*[contains(@class, 'caption')]")
MW_DROPPED_CLASSES = (
    "mw-editsection",
    "reference",
    "navbox",
    "navbox-styles",
    "vertical-navbox",
    "sidebar",
    "ambox",
    "hatnote",
    "noprint",
    "metadata",
    "toc",
    "shortdescription",
    "mw-empty-elt",
    "portalbox",
    "sistersitebox",
    "side-box",
    "mw-cite-backlink",
    "error",
)
MW_DROPPED = etree.XPath(
    "//*["
    + " or ".join(
        f"contains(concat(' ', normalize-space(@class), ' '), ' {c} ')"
        for c in MW_DROPPED_CLASSES
    )
    + " or @role='navigation' or @id='toc' or @role='note']"
)
MW_SKIPPED_SECTIONS = frozenset(("see also", "external links"))


def is_mediawiki(root: etree._Element, generator: str) -> bool:
    return generator.lower().startswith("mediawiki") and bool(MW_CONTENT(root))


def mediawiki_text(
    root: etree._Element, title: str, cap: int = MAX_TEXT_CHARS
) -> tuple[str, list[str]]:
    """A MediaWiki article: short description, lead, infobox as `Field: value` lines, sections; and its categories."""
    tree = deepcopy(root)
    categories = [inline_text(a) for a in MW_CATEGORIES(tree)]
    short = MW_SHORT(tree)
    lines = [Line(inline_text(short[0]))] if short and inline_text(short[0]) else []
    body = MW_CONTENT(tree)[0]
    etree.strip_elements(body, *MAIN_DROPPED, with_tail=False)
    infobox: list[Line] = []
    for box in MW_INFOBOXES(body):
        for mark in MW_MARKS(box):
            if mark.getparent() is not None:
                mark.drop_tree()
        infobox += infobox_lines(box)
        box.drop_tree()
    for node in MW_DROPPED(body):
        if node.getparent() is not None:
            node.drop_tree()
    for figure in body.iter("figure"):
        caption = next(iter(figure.iter("figcaption")), None)
        text = inline_text(caption) if caption is not None else ""
        for child in list(figure):
            figure.remove(child)
        figure.text = text
    lead: list[Line] = []
    sections: list[Line] = []
    skipping: int | None = None
    in_sections = False
    for child in sections_of(body):
        heading = heading_of(child)
        if heading is not None:
            in_sections = True
            level = HEADINGS[tag_of(heading)]
            if skipping is not None and level <= skipping:
                skipping = None
            if inline_text(heading).lower() in MW_SKIPPED_SECTIONS:
                skipping = level
                continue
            child = heading
        if skipping is not None:
            continue
        (sections if in_sections else lead).extend(render(child))
    labels = [category for category in categories if category]
    return finish(title, clean(lines + lead + infobox + sections), cap), labels


def infobox_lines(box: etree._Element) -> list[Line]:
    lines = []
    for row in box.iter("tr"):
        label = next((c for c in row if tag_of(c) == "th"), None)
        cells = [c for c in row if tag_of(c) == "td"]
        value = cells[0] if cells else None
        if label is not None and value is not None:
            name = inline_text(label)
            text = ", ".join(t for t in (inline_text(c, ", ") for c in cells) if t)
            if name and text:
                lines.append(Line(f"{name}: {text}"))
        elif value is not None and len(cells) == 1:
            if value.findall(".//img"):
                text = " ".join(t for t in map(inline_text, captions(value)) if t)
            else:
                text = inline_text(value, ", ")
            if text:
                lines.append(Line(text))
        elif label is not None and inline_text(label):
            lines.append(Line(inline_text(label)))
    return lines


def captions(cell: etree._Element) -> list[etree._Element]:
    """The outermost caption elements of an image cell, like "infobox-caption"."""
    found = MW_CAPTIONS(cell)
    return [c for c in found if not any(a in found for a in c.iterancestors())]


def sections_of(node: etree._Element):
    for child in node:
        if tag_of(child) == "section":
            yield from sections_of(child)
        elif tag_of(child):
            yield child


def heading_of(node: etree._Element) -> etree._Element | None:
    if tag_of(node) in HEADINGS:
        return node
    if "mw-heading" in (node.get("class") or ""):
        return next((c for c in node if tag_of(c) in HEADINGS), None)
    return None
