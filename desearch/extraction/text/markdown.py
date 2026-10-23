"""HTML rendered as markdown lines: blocks, lists, tables and code."""

import re
from typing import NamedTuple

from lxml import etree

from desearch.extraction.text.dom import (
    BLOCKS,
    HEADINGS,
    alnum,
    chars,
    inline_text,
    link_only,
    opens_with_punctuation,
    spaced,
    tag_of,
    tidy,
)

MAX_TABLE_ROWS = 2000
MAX_TABLE_COLUMNS = 50
MAX_SPAN = 50
MAX_ITEM_CHARS = 1000
MAX_TERM_CHARS = 80
MAX_PAIR_CHARS = 300
NUMBER = re.compile(r"^-?\d{1,6}$")
CELLS = ("td", "th")
LISTS = ("ul", "ol")
LIST_MARK = re.compile(r"^\s*(-|\d+\.) ")


class Line(NamedTuple):
    text: str
    linked: bool = False
    verbatim: bool = False
    row: bool = False


class Renderer:
    def __init__(self, cards: bool = False) -> None:
        self.lines: list[Line] = []
        self.buffer: list[tuple[str, bool]] = []
        self.link_depth = 0
        self.cards = cards

    def flush(self) -> None:
        text = tidy("".join(s for s, _ in self.buffer))
        inside = sum(alnum(s) for s, linked in self.buffer if linked)
        outside = sum(alnum(s) for s, linked in self.buffer if not linked)
        self.buffer = []
        if text:
            self.lines.append(Line(text, inside > 0 and outside == 0))

    def add(self, text: str | None) -> None:
        if text:
            self.buffer.append((text, self.link_depth > 0))

    def walk(self, node: etree._Element, indent: str = "") -> None:
        tag = tag_of(node)
        if not tag:
            return
        if tag in HEADINGS:
            self.flush()
            heading = inline_text(node)
            if heading:
                self.lines.append(
                    Line(
                        f"{'#' * HEADINGS[tag]} {heading}",
                        link_only(node) or self.link_depth > 0,
                    )
                )
            return
        if tag == "table":
            self.flush()
            self.table(node)
            return
        if tag in LISTS:
            self.flush()
            self.items(node, indent)
            return
        if tag == "dl":
            self.flush()
            self.definitions(node, indent)
            return
        if tag == "pre":
            self.flush()
            code = "".join(node.itertext()).strip("\n")
            if code.strip():
                self.lines.append(Line(f"```\n{code}\n```", verbatim=True))
            return
        if tag == "blockquote":
            self.flush()
            inner = Renderer()
            inner.children(node, indent)
            inner.flush()
            self.lines.extend(
                Line(f"> {line.text}", line.linked, line.verbatim)
                for line in inner.lines
            )
            return
        block = tag in BLOCKS
        if block or tag == "br":
            self.flush()
        if spaced(node) and not opens_with_punctuation(node):
            self.add(" ")
        if tag == "a":
            self.link_depth += 1
        self.children(node, indent)
        if tag == "a":
            self.link_depth -= 1
        if spaced(node):
            self.add(" ")
        if block:
            self.flush()

    def children(self, node: etree._Element, indent: str) -> None:
        self.add(node.text)
        for child in node:
            self.walk(child, indent)
            self.add(child.tail)

    def items(self, node: etree._Element, indent: str) -> None:
        ordered = tag_of(node) == "ol"
        entries = [c for c in node if tag_of(c) == "li"]
        numbers = list_numbers(node, entries) if ordered else []
        for i, item in enumerate(entries):
            bullet = f"{numbers[i]}." if ordered else "-"
            if chars(item) > MAX_ITEM_CHARS or (self.cards and sectioned(item)):
                inner = Renderer(self.cards)
                inner.children(item, indent)
                inner.flush()
                if (
                    inner.lines
                    and not inner.lines[0].verbatim
                    and not inner.lines[0].row
                    and not LIST_MARK.match(inner.lines[0].text)
                ):
                    first = inner.lines[0]
                    inner.lines[0] = first._replace(
                        text=f"{indent}{bullet} {first.text.lstrip('# ')}"
                    )
                self.lines.extend(inner.lines)
                continue
            line = inline_text(item, skip=LISTS)
            if line:
                self.lines.append(
                    Line(
                        f"{indent}{bullet} {line}",
                        link_only(item)
                        and not item.findall(".//ul")
                        and not item.findall(".//ol"),
                    )
                )
            # A nested list starts where its parent item's text starts, as CommonMark reads it.
            for sub in nested_lists(item):
                self.items(sub, indent + " " * (len(bullet) + 1))

    def definitions(self, node: etree._Element, indent: str) -> None:
        for term, values in definition_pairs(node):
            label = inline_text(term) if term is not None else ""
            simple = [
                inline_text(v)
                for v in values
                if not any(tag_of(d) in BLOCKS for d in v.iterdescendants())
            ]
            if (
                label
                and values
                and len(simple) == len(values)
                and len(label) <= MAX_TERM_CHARS
                and sum(map(len, simple)) <= MAX_PAIR_CHARS
            ):
                self.lines.append(
                    Line(
                        f"{indent}{label}: {'; '.join(t for t in simple if t)}",
                        link_only(term) and all(link_only(v) for v in values),
                    )
                )
                continue
            if label:
                self.lines.append(Line(f"{indent}{label}", link_only(term)))
            for value in values:
                inner = Renderer(self.cards)
                inner.children(value, indent)
                inner.flush()
                self.lines.extend(inner.lines)

    def table(self, table: etree._Element) -> None:
        caption = next((c for c in table if tag_of(c) == "caption"), None)
        if caption is not None and inline_text(caption):
            self.lines.append(Line(inline_text(caption)))
        grid = table_grid(table)
        if not grid:
            return
        if max(len(cells) for cells, _ in grid) <= 1:
            self.lines.extend(Line(cell) for cells, _ in grid for cell in cells if cell)
            return
        for i, (cells, header) in enumerate(grid):
            if any(cells):
                self.lines.append(Line("| " + " | ".join(cells) + " |", row=True))
                if i == 0 and header:
                    self.lines.append(Line("|" + "---|" * len(cells), row=True))


def list_numbers(node: etree._Element, entries: list[etree._Element]) -> list[int]:
    """The number of each item of an ordered list: its start, explicit item values, counting down when reversed."""
    step = -1 if node.get("reversed") is not None else 1
    start = node.get("start", "")
    number = int(start) if NUMBER.match(start) else (len(entries) if step < 0 else 1)
    numbers = []
    for item in entries:
        value = item.get("value", "")
        if NUMBER.match(value):
            number = int(value)
        numbers.append(number)
        number += step
    return numbers


def definition_pairs(
    node: etree._Element,
) -> list[tuple[etree._Element | None, list[etree._Element]]]:
    """A description list's terms, each with its definitions; <div> groups inside the list are read through."""
    pairs: list[tuple[etree._Element | None, list[etree._Element]]] = []
    for child in node.iter("dt", "dd"):
        if next(a for a in child.iterancestors() if tag_of(a) == "dl") is not node:
            continue
        if tag_of(child) == "dt":
            pairs.append((child, []))
        elif pairs:
            pairs[-1][1].append(child)
        else:
            pairs.append((None, [child]))
    return pairs


def sectioned(item: etree._Element) -> bool:
    """A list item laid out as a card with a heading of its own."""
    return any(tag_of(n) in HEADINGS for n in item.iter())


def nested_lists(node: etree._Element):
    """The lists inside a list item, however deep, but not lists inside those."""
    for child in node:
        if tag_of(child) in LISTS:
            yield child
        elif tag_of(child):
            yield from nested_lists(child)


def span(cell: etree._Element, name: str) -> int:
    value = (cell.get(name) or "").strip()
    return min(int(value), MAX_SPAN) if value.isdecimal() and int(value) > 0 else 1


def nearest_table(node: etree._Element) -> etree._Element | None:
    return next((a for a in node.iterancestors() if tag_of(a) == "table"), None)


def table_grid(table: etree._Element) -> list[tuple[list[str], bool]]:
    """Rows of cell texts with row and column spans copied into every cell they cover, and whether a row is all headers."""
    rows = [r for r in table.iter("tr") if nearest_table(r) is table][:MAX_TABLE_ROWS]
    grid: list[tuple[list[str], bool]] = []
    carried: dict[int, tuple[str, int]] = {}
    for row in rows:
        cells = [c for c in row if tag_of(c) in CELLS]
        out: list[str] = []
        if len(cells) == 1 and not carried and span(cells[0], "colspan") > 1:
            width = span(cells[0], "colspan")
            grid.append(
                (
                    [inline_text(cells[0], "; ")] + [""] * (width - 1),
                    tag_of(cells[0]) == "th",
                )
            )
            continue
        queue = iter(cells)
        while len(out) < MAX_TABLE_COLUMNS:
            column = len(out)
            if column in carried:
                text, left = carried.pop(column)
                out.append(text)
                if left > 1:
                    carried[column] = (text, left - 1)
                continue
            cell = next(queue, None)
            if cell is None:
                break
            text = inline_text(cell, "; ")
            down = span(cell, "rowspan")
            for _ in range(span(cell, "colspan")):
                if down > 1:
                    carried[len(out)] = (text, down - 1)
                out.append(text)
        grid.append(
            (
                out[:MAX_TABLE_COLUMNS],
                bool(cells) and all(tag_of(c) == "th" for c in cells),
            )
        )
    return grid


def render(node: etree._Element, cards: bool = False) -> list[Line]:
    renderer = Renderer(cards)
    renderer.walk(node)
    renderer.flush()
    return renderer.lines


def heading_level(line: Line) -> int:
    marks = len(line.text) - len(line.text.lstrip("#"))
    return marks if 0 < marks <= 6 and line.text[marks : marks + 1] == " " else 0


def with_title(title: str, lines: list[str]) -> list[str]:
    first = lines[0].lstrip("#").strip() if lines else ""
    if title and first.casefold() != title.casefold():
        return [title, *lines]
    return lines


def finish(title: str, lines: list[Line], cap: int) -> str:
    text = "\n".join(with_title(title, [line.text for line in lines]))
    if len(text) <= cap:
        return text
    cut = text.rfind("\n", 0, cap)
    return text[: cut if cut > 0 else cap]
