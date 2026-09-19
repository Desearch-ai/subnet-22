"""Enough of a homepage to tell whether it is readable English."""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass

MIN_CHARS = 200
SAMPLE = 2000
HTML_LANG = re.compile(r"<html[^>]*\blang=[\"']?([a-zA-Z-]{2,8})", re.I)
SCRIPT_OR_STYLE = re.compile(r"<(script|style|noscript|svg)[^>]*>.*?</\1>", re.S | re.I)
TAG = re.compile(r"<[^>]+>")
CHARSET = re.compile(rb'charset=["\']?([\w-]+)', re.I)
BOT_WALL = re.compile(
    r"(please enable javascript|enable javascript and refresh|you need to enable javascript"
    r"|access denied|are you a robot|checking your browser|attention required)",
    re.I,
)


@dataclass(frozen=True)
class Homepage:
    chars: int
    declared: str | None
    language: str | None
    problem: str | None


def read(body: bytes, detect_language: Callable[[str], str | None]) -> Homepage:
    """Judge a homepage; problem is None when it is readable English."""
    html = decode(body)
    match = HTML_LANG.search(html[:4000])
    declared = match.group(1).lower().split("-")[0] if match else None
    text = " ".join(TAG.sub(" ", SCRIPT_OR_STYLE.sub(" ", html)).split())
    if BOT_WALL.search(text[:SAMPLE]):
        return Homepage(len(text), declared, None, "bot_wall")
    if len(text) < MIN_CHARS:
        return Homepage(len(text), declared, None, "no_text")
    language = detect_language(text[:SAMPLE])
    return Homepage(
        len(text), declared, language, None if language == "en" else "not_english"
    )


def decode(body: bytes) -> str:
    match = CHARSET.search(body[:4096])
    encoding = match.group(1).decode("ascii", "ignore") if match else "utf-8"
    try:
        return body.decode(encoding, "replace")
    except LookupError:
        return body.decode("utf-8", "replace")
