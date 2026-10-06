"""Writes src/entities.rs from Python's own table of HTML5 named character references."""

import sys
from html.entities import html5
from pathlib import Path


def literal(text: str) -> str:
    out = []
    for ch in text:
        if ch.isascii() and ch.isprintable() and ch not in '"\\':
            out.append(ch)
        else:
            out.append(f"\\u{{{ord(ch):x}}}")
    return '"' + "".join(out) + '"'


def main() -> None:
    target = Path(sys.argv[1] if len(sys.argv) > 1 else Path(__file__).parents[1] / "src" / "entities.rs")
    lines = [
        "//! HTML5 named character references as Python's `html.entities.html5` lists them, written by tools/entities.py.",
        "",
        "/// Sorted by name, with and without the closing semicolon where the standard allows both.",
        "pub const ENTITIES: &[(&str, &str)] = &[",
    ]
    for name in sorted(html5, key=lambda n: n.encode()):
        lines.append(f"    ({literal(name)}, {literal(html5[name])}),")
    lines.append("];")
    target.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
