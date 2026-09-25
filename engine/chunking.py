"""Article -> index units: paragraph chunks + two document-level texts (head, full).

Every character of the article body reaches some chunk: short paragraphs merge with their
neighbours rather than being dropped, long ones split rather than being truncated. Dropping
sub-120-char paragraphs silently removed 14.2% of corpus text across 36% of paragraphs.
"""

from __future__ import annotations

import re

CHUNK_MIN, CHUNK_MAX = 120, 1600
HEAD_CHARS, FULL_CHARS = 1500, 8000

_SENT = re.compile(r"(?<=[.!?])\s+")


def _split_long(p: str) -> list[str]:
    """Pack sentences up to CHUNK_MAX; a single oversized sentence falls back to a hard slice."""
    out, buf = [], ""
    for s in _SENT.split(p):
        if len(s) > CHUNK_MAX:
            if buf:
                out.append(buf)
                buf = ""
            out += [s[i : i + CHUNK_MAX] for i in range(0, len(s), CHUNK_MAX)]
            continue
        if buf and len(buf) + 1 + len(s) > CHUNK_MAX:
            out.append(buf)
            buf = s
        else:
            buf = f"{buf} {s}" if buf else s
    if buf:
        out.append(buf)
    return out


def para_chunks(text: str) -> list[str]:
    out, buf = [], ""
    for p in re.split(r"\n{1,}", text or ""):
        p = p.strip()
        if not p:
            continue
        if len(p) > CHUNK_MAX:
            if buf:
                out.append(buf)
                buf = ""
            out += _split_long(p)
            continue
        buf = f"{buf}\n{p}" if buf else p
        if len(buf) >= CHUNK_MIN:
            out.append(buf)
            buf = ""
    if buf:
        # trailing shorts have no neighbour left to merge into; keep them rather than lose them
        out.append(buf)
    return out


def doc_head(title: str, text: str) -> str:
    return (title or "") + "\n" + (text or "")[:HEAD_CHARS]


def doc_full(title: str, text: str) -> str:
    return (title or "") + "\n" + (text or "")[:FULL_CHARS]
