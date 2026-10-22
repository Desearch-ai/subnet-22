"""Line cleanup after rendering: repeats, junk lines, and what ends an article after its body."""

import re

from desearch.extraction.text.dom import MAX_CONTROL_WORDS, VALUE, WORDS, alnum
from desearch.extraction.text.markdown import Line, heading_level

MIN_LINE_CHARS = 2
MIN_REPEAT_WORDS = 8
MIN_LINK_RUN = 8
MAX_LINK_RUN_WORDS = 5
MIN_TRAILING_LINKS = 2
BLOB = re.compile(r"\S{300,}")
SKIP_LINK = re.compile(
    r"^skip to (the )?(main )?(content|navigation|search|product( information)?)\b",
    re.I,
)
RELATED_LABEL = re.compile(
    r"^(related( (articles?|stories|story|posts?|content|coverage|news|reading|links|topics|videos?))?"
    r"|related internet links|more|see more( from .{1,40})?|top news"
    r"|more (on|about) this( story| topic)?|more (to read|stories|news)"
    r"|more (from|by|in) .{1,40}|read (more|next|also)( from .{1,40})?|also read|see also"
    r"|recommended( (reading|stories|for you|articles|videos))?|you (may|might) (also )?like"
    r"|most (read|popular|viewed|shared)|trending( now| stories)?|popular( stories| now)?|top stories"
    r"|in case you missed it|don.?t miss|up next|what to read next|editor.?s picks|sponsored picks"
    r"|filed under|topics|tags)(?<![.!?])\s*:?$",
    re.I,
)
COPYRIGHT = re.compile(r"^(©|\(c\)|copyright\b)|all rights reserved", re.I)
RELATED_LEAD = re.compile(
    r"^(read more|related|see also|also read|read next|more)\s*:\s+\S", re.I
)
CALL_TO_ACTION = re.compile(
    r"^(follow (us|me|@|topics|along)|subscribe|sign up|sign in|log in|register|donate|support (local|our|independent)"
    r"|become a (member|subscriber|supporter)|get (the|our) (app|newsletter)|download (the|our) app"
    r"|listen to (this|the) (article|story)|this (audio|article) (is|was) (auto|generated|read)"
    r"|add .{1,40}(google|preferred source)|want more|don.?t miss)\b",
    re.I,
)
GENERIC_ACTION = re.compile(
    r"^(more|more info(rmation)?|more details|details|view details|read more|learn more|see more|view more"
    r"|show more|load more|find out more|discover more|explore( all| more)?|view all|see all|show all"
    r"|browse all|continue( reading)?|click here|get started|start now|shop now|buy now|order now"
    r"|book now|apply now|join now|watch now)\W*$",
    re.I,
)
AUDIO_PLAYER = re.compile(
    r"^(getting your .{0,30}audio player ready|click here to listen to this (article|story)"
    r"|(this|the) (audio|article|story) is (auto-generated|read by an automated voice)"
    r"|this is read by an automated voice|please report any issues or inconsistencies here)",
    re.I,
)
BIO_ROLE = re.compile(
    r"\b(reporter|writer|editor|editor-in-chief|correspondent|columnist|contributor|journalist|journalism"
    r"|producer|strategist|analyst|fellow|critic|photographer|intern)s?\b|\b(writes|reports|covers) (about|on|for|the)\b",
    re.I,
)
BIO_PRONOUN = re.compile(r"^(he|she|they|his|her|their)\b", re.I)
BIO_CONTACT = re.compile(
    r"^(please )?(you can )?(reach( out to)?|contact|email|follow|connect with)\b.{0,60}(@|\blinkedin\b|\btwitter\b|\bemail\b)"
    r"|^more (by|from) (?-i:[A-Z])",
    re.I,
)
MAX_BIO_CHARS = 600
NAME_FILLERS = frozenset(("The", "And", "For", "With", "From", "Staff", "Team", "News"))
INTERFACE_LINE = re.compile(
    r"^(you are using an? (out[- ]of[- ]date|unsupported|outdated) browser|you should upgrade or use an alternative browser"
    r"|it seems that you.?re using an outdated browser|please use .{1,20} browser for"
    r"|an official website of the united states|a \.gov website belongs to an official|share sensitive information only on official"
    r"|double tap to zoom|use (the )?left/right arrows to navigate|listen to (this|the) (article|story)\W*$"
    r"|affiliate disclosure\b|this (article|post|page) (may )?contains? affiliate links"
    r"|when you (buy|purchase|shop) (through|via|using) links (on|in) our"
    r"|(website|web design|site) by\b|government websites by\b|powered by wordpress)",
    re.I,
)
GALLERY_COUNTER = re.compile(r"^(image|photo|slide|item|picture) \d+ (of|/) \d+$", re.I)
APPEAL = re.compile(
    r"^(support (us|our|the)\b|make a (one-off |one-time |monthly |recurring )?donation|keep (our|the) (news|journalism|site)\b"
    r"|read .{1,30} ad-free|remove ads and support|this content is reserved for subscribers|create an account or sign in"
    r"|unlock subscriber|become a (member|patron|supporter)|you.?re reading .{1,40} brought to you by"
    r"|safeguard press freedom|click here to help us)",
    re.I,
)
MAX_TAIL_WORDS = 40
MAX_INTERFACE_WORDS = 30
JSON_TEXT = re.compile(r'^[\[{]+\s*"[^"]{1,100}"\s*:')
AD_LABEL = re.compile(
    r"^(advertisements?|sponsored( content)?|paid (post|content)|story continues below( advertisement)?)( \d+)?$",
    re.I,
)


def repeatable(line: Line) -> bool:
    """A short label or value that sibling cards can each hold, like a price, a plan feature or an author."""
    return len(line.text.split()) < MIN_REPEAT_WORDS


def clean(lines: list[Line], cards: bool = False) -> list[Line]:
    """Drops skip links, blobs, symbol-only and tiny lines, repeated tables and repeated lines, except, with `cards`, the short labels and values sibling cards repeat."""
    seen: set[str] = set()
    tables: set[tuple[str, ...]] = set()
    kept: list[Line] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.row:
            j = i
            while j < len(lines) and lines[j].row:
                j += 1
            rows = tuple(row.text for row in lines[i:j])
            if not cards or rows not in tables:
                tables.add(rows)
                kept.extend(lines[i:j])
            i = j
            continue
        i += 1
        if line.verbatim:
            kept.append(line)
            continue
        bare = line.text.strip("-#|> ")
        if (
            (len(bare) <= MIN_LINE_CHARS and not (cards and VALUE.search(bare)))
            or (cards and not alnum(bare))
            or SKIP_LINK.match(bare)
            or AD_LABEL.match(bare)
            or JSON_TEXT.match(bare)
            or AUDIO_PLAYER.match(bare)
            or BLOB.search(line.text)
            or (
                COPYRIGHT.match(bare)
                and (cards or len(bare.split()) <= MAX_CONTROL_WORDS * 2)
            )
            or (len(bare.split()) <= MAX_INTERFACE_WORDS and INTERFACE_LINE.match(bare))
            or GALLERY_COUNTER.match(bare)
            or (
                cards
                and len(bare.split()) < MIN_REPEAT_WORDS
                and CALL_TO_ACTION.match(bare)
            )
            or (cards and GENERIC_ACTION.match(bare))
            or (line.text in seen and not (cards and repeatable(line)))
        ):
            continue
        seen.add(line.text)
        kept.append(line)
    return kept


def drop_author_bio(lines: list[Line], author: str) -> list[Line]:
    """What ends an article after its body: calls to action and appeals, and the author's bio and contact lines, found by the byline's names."""
    names = {
        w
        for w in WORDS.findall(author)
        if len(w) >= 3 and w[0].isupper() and w not in NAME_FILLERS
    }
    while True:
        end = len(lines)
        while end and tail_action(lines[end - 1].text.strip("-#> ")):
            end -= 1
        stop, named = end, False
        while stop and names:
            text = lines[stop - 1].text.strip("-#> ")
            if len(text) > MAX_BIO_CHARS:
                break
            mentions = bool(names & set(WORDS.findall(text)))
            if BIO_ROLE.search(text) and (mentions or BIO_PRONOUN.match(text)):
                named = named or mentions
            elif not BIO_CONTACT.match(text):
                break
            stop -= 1
        if named:
            end = stop
        if end == len(lines):
            return lines
        lines = lines[:end]


def tail_action(text: str) -> bool:
    return len(text.split()) <= MAX_TAIL_WORDS and bool(
        CALL_TO_ACTION.match(text) or APPEAL.match(text)
    )


def drop_orphans(lines: list[Line]) -> list[Line]:
    """Drops related-links labels with only links under them, the headings and link runs that end the text, and links before the headline."""
    kept: list[Line] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if not line.verbatim and RELATED_LEAD.match(line.text.lstrip("-#> ")):
            i += 1
            continue
        if not line.verbatim and RELATED_LABEL.match(line.text.lstrip("# ")):
            j = i + 1
            while j < len(lines) and lines[j].linked:
                j += 1
            if j > i + 1 or j == len(lines) or heading_level(lines[j]):
                i = j
                continue
        kept.append(line)
        i += 1
    out = list(kept)
    while out:
        links = 0
        while (
            links < len(out)
            and out[-1 - links].linked
            and not heading_level(out[-1 - links])
        ):
            links += 1
        if links >= MIN_TRAILING_LINKS:
            del out[-links:]
        elif heading_level(out[-1]):
            out.pop()
        else:
            break
    first = next((i for i, line in enumerate(out) if line.text.startswith("# ")), 0)
    return [line for i, line in enumerate(out) if i >= first or not line.linked]


def drop_link_runs(lines: list[Line]) -> list[Line]:
    """Drops runs of short link-only lines, the menus and link directories left in a page, but not cards with text."""
    out: list[Line] = []
    run: list[Line] = []
    for line in [*lines, Line("")]:
        if (
            line.linked
            and not line.text.startswith("#")
            and len(line.text.split()) <= MAX_LINK_RUN_WORDS
        ):
            run.append(line)
            continue
        if len(run) < MIN_LINK_RUN:
            out.extend(run)
        run = []
        if line.text:
            out.append(line)
    return out
