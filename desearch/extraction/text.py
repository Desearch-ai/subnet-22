from copy import deepcopy

import trafilatura
from lxml import etree

MIN_TEXT_CHARS = 200

HIDDEN_TAGS = (
    "script",
    "style",
    "noscript",
    "template",
    "svg",
    "nav",
    "header",
    "footer",
    "aside",
    "form",
    "iframe",
    "button",
    "select",
    "dialog",
)
BLOCK_TAGS = (
    "p",
    "div",
    "li",
    "ul",
    "ol",
    "h1",
    "h2",
    "h3",
    "h4",
    "h5",
    "h6",
    "tr",
    "td",
    "th",
    "table",
    "section",
    "article",
    "main",
    "br",
    "blockquote",
    "pre",
    "dd",
    "dt",
    "dl",
    "figcaption",
    "figure",
    "address",
    "hr",
)
HIDDEN_NODES = etree.XPath(
    "//*[@hidden or @aria-hidden='true' or @role='navigation' or @role='banner'"
    " or @role='contentinfo' or contains(@id, 'cookie') or contains(@class, 'cookie')"
    " or contains(@id, 'consent') or contains(@class, 'consent')]"
)


def tidy(text: str) -> str:
    lines = (" ".join(line.split()) for line in (text or "").splitlines())
    return "\n".join(line for line in lines if line)


def trafilatura_text(root: etree._Element, precision: bool) -> str:
    try:
        text = trafilatura.extract(
            root,
            fast=True,
            favor_precision=precision,
            favor_recall=not precision,
            include_comments=False,
            include_tables=True,
            deduplicate=False,
        )
    except Exception:
        return ""
    return tidy(text or "")


def visible_text(root: etree._Element) -> str:
    body = root.find("body")
    tree = deepcopy(body if body is not None else root)
    etree.strip_elements(tree, *HIDDEN_TAGS, with_tail=False)
    for node in HIDDEN_NODES(tree):
        if node.getparent() is not None:
            node.drop_tree()
    for node in tree.iter(*BLOCK_TAGS):
        node.tail = "\n" + (node.tail or "")
    return tidy(tree.text_content())


def main_text(root: etree._Element, page_type: str) -> str:
    modes = (True, False) if page_type == "article" else (False,)
    best = ""
    for precision in modes:
        text = trafilatura_text(root, precision)
        if len(text) >= MIN_TEXT_CHARS:
            return text
        best = max(best, text, key=len)
    fallback = visible_text(root)
    return fallback if len(fallback) > len(best) else best
