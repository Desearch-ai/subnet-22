"""A URL read as the classifier needs it: path segments without locale or file suffix, query pairs, home and archive paths."""

import re
from urllib.parse import parse_qsl, unquote, urlsplit

LOCALES = {
    "en",
    "fr",
    "de",
    "es",
    "it",
    "nl",
    "pt",
    "pl",
    "ru",
    "ja",
    "zh",
    "ko",
    "sv",
    "da",
    "no",
    "nb",
    "fi",
    "tr",
    "ar",
    "he",
    "cs",
    "hu",
    "ro",
    "el",
    "uk",
    "id",
    "vi",
    "th",
}
INDEX_FILES = {"index", "index.html", "index.htm", "index.php", "home", "default.aspx"}
FILE_SUFFIX = re.compile(r"\.(html?|php|aspx?|jsp|cfm)$")
REGIONAL_LOCALE = re.compile(r"^[a-z]{2}[-_][a-z]{2}$")
PAGE_KEYS = {
    "page",
    "paged",
    "pg",
    "pgno",
    "start",
    "offset",
    "p",
    "pagenum",
    "page_no",
}
PAGE_SEGMENT = re.compile(r"^(page[-_]?\d+|\d+\.page)$")
PAGED_PATH_KEYS = re.compile(r"(^|&)(page|paged|pg)=\d+")
YEAR = re.compile(r"^(19|20)\d\d$")


def path_segments(url: str) -> list[str]:
    try:
        parts = urlsplit(url or "")
    except ValueError:
        return []
    segments = [s for s in unquote(parts.path).lower().split("/") if s.strip()]
    if segments and (segments[0] in LOCALES or REGIONAL_LOCALE.match(segments[0])):
        segments = segments[1:]
    return [
        FILE_SUFFIX.sub("", s) if i == len(segments) - 1 else s
        for i, s in enumerate(segments)
    ]


def words(segment: str) -> list[str]:
    return [w for w in re.split(r"[-_+.\s]+", segment) if w]


def query_pairs(url: str) -> list[tuple[str, str]]:
    try:
        query = urlsplit(url).query
    except ValueError:
        return []
    return [(k.lower(), v) for k, v in parse_qsl(query, keep_blank_values=True)]


def is_home(segments: list[str]) -> bool:
    return not segments or (len(segments) == 1 and segments[0] in INDEX_FILES)


def is_date_archive(segments: list[str]) -> bool:
    """A path that ends in a year and month, or a year, month and day: /blog/2021/08/."""
    return any(
        len(segments) >= size
        and bool(YEAR.match(segments[-size]))
        and all(s.isdecimal() and len(s) <= 2 for s in segments[1 - size :])
        for size in (2, 3)
    )


def is_paginated(url: str, segments: list[str], pairs: list[tuple[str, str]]) -> bool:
    if any(
        s == "page" and segments[i + 1].isdecimal() for i, s in enumerate(segments[:-1])
    ):
        return True
    if any(PAGE_SEGMENT.match(s) for s in segments):
        return True
    query = urlsplit(url).query.lower()
    return bool(PAGED_PATH_KEYS.search(query)) or any(
        k in PAGE_KEYS and v.isdecimal() and int(v) > 1 for k, v in pairs
    )
