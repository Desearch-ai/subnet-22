"""arXiv: an abstract page per paper from its CC0 metadata (OAI-PMH pages or the Kaggle snapshot), and full text for openly licensed papers from Common Pile."""

from __future__ import annotations

import gzip
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from collections.abc import Iterator
from datetime import datetime
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import NamedTuple

from lxml import etree

from bulk.rows import Batches, authors_line, paper_page, upload_row
from desearch.extraction.text.markdown import Line

OAI = "https://oaipmh.arxiv.org/oai"
NS = {"o": "http://www.openarchives.org/OAI/2.0/", "a": "http://arxiv.org/OAI/arXiv/"}
# arXiv's robots.txt asks for 15 s between requests; OAI-PMH asks for 3 s and sends Retry-After when busy.
DELAY_S = 15.0
SITE = "arXiv"
# Licences under which a commercial index may keep and show a paper's full text.
OPEN = frozenset(
    (
        "creativecommons.org/licenses/by/4.0/",
        "creativecommons.org/licenses/by/3.0/",
        "creativecommons.org/licenses/by-sa/4.0/",
        "creativecommons.org/publicdomain/zero/1.0/",
        "creativecommons.org/licenses/publicdomain/",
    )
)
# arXiv makes an HTML view of every TeX submission from December 2023 on; ar5iv holds the view of earlier papers.
HTML_FROM = "2312"
# Shorter full texts are failed conversions; the abstract page covers the paper.
MIN_TEXT_CHARS = 2_000
REFERENCES = frozenset(
    ("references", "bibliography", "references and notes", "literature cited")
)
HEADING = re.compile(r"^(#{1,6})(?:\s+(.*?))?\s*$")
NUMBERING = re.compile(r"^(\d+(\.\d+)*|[ivxlc]+)\.?\s+")
ITEM = re.compile(r"^\s*([*+-]|\d+\.)\s")
ABSTRACT = re.compile(r"^\s*abstract\b", re.IGNORECASE)
EMAIL = re.compile(r"[\w.+-]+@[\w-]+(\.[\w-]+)+")
# A short paragraph with an email address is an affiliation or correspondence line.
AFFILIATION_CHARS = 500
ESCAPED = re.compile(r"\\([.*_#\[\]()+!-])")


class Paper(NamedTuple):
    arxiv_id: str
    title: str
    authors: list[str]
    abstract: str
    categories: str
    created: str
    updated: str
    comments: str
    journal: str
    doi: str
    licence: str


def url_of(arxiv_id: str) -> str:
    return f"https://arxiv.org/abs/{arxiv_id}"


def html_url(arxiv_id: str) -> str:
    """The HTML view that shows a paper's full text."""
    if "/" not in arxiv_id and arxiv_id[:4] >= HTML_FROM:
        return f"https://arxiv.org/html/{arxiv_id}"
    return f"https://ar5iv.labs.arxiv.org/html/{arxiv_id}"


def is_open(licence: str) -> bool:
    return licence.strip().split("://", 1)[-1] in OPEN


def squash(text: str | None) -> str:
    return " ".join((text or "").split())


def harvest(out: Path, since: str, until: str, user_agent: str) -> int:
    """Saves every OAI-PMH page of records changed from `since` to `until` (dates) as numbered XML files; returns the pages saved."""
    out.mkdir(parents=True, exist_ok=True)
    query = {
        "verb": "ListRecords",
        "metadataPrefix": "arXiv",
        "from": since,
        "until": until,
    }
    pages = 0
    while True:
        request = urllib.request.Request(
            f"{OAI}?{urllib.parse.urlencode(query)}", headers={"User-Agent": user_agent}
        )
        try:
            with urllib.request.urlopen(request, timeout=300) as response:
                body = response.read()
        except urllib.error.HTTPError as error:
            if error.code in (429, 503):
                time.sleep(float(error.headers.get("Retry-After") or 60))
                continue
            raise
        pages += 1
        (out / f"oai-{pages:06d}.xml").write_bytes(body)
        token = etree.fromstring(body).find(".//o:resumptionToken", NS)
        if token is None or not (token.text or "").strip():
            return pages
        query = {"verb": "ListRecords", "resumptionToken": token.text.strip()}
        time.sleep(DELAY_S)


def heading_lines(paper: Paper) -> list[Line]:
    stamp = f"Submitted {paper.created}" + (
        f", updated {paper.updated}"
        if paper.updated and paper.updated != paper.created
        else ""
    )
    return [
        Line(authors_line(paper.authors)),
        Line(
            " · ".join(
                p for p in (f"arXiv:{paper.arxiv_id}", paper.categories, stamp) if p
            )
        ),
    ]


def framed(paper: Paper, url: str, lines: list[Line]):
    return url, paper_page(
        paper.title,
        lines,
        lang="en",
        url=url,
        published=paper.created,
        modified=paper.updated if paper.updated != paper.created else "",
        author=authors_line(paper.authors),
        site_name=SITE,
        tags=paper.categories.split(),
    )


def page_of(paper: Paper):
    lines = heading_lines(paper) + [Line("## Abstract"), Line(paper.abstract)]
    lines += [
        Line(f"{name}: {value}")
        for name, value in (
            ("Comments", paper.comments),
            ("Journal reference", paper.journal),
            ("DOI", paper.doi),
        )
        if value
    ]
    return framed(paper, url_of(paper.arxiv_id), lines)


def body_lines(markdown: str) -> list[str]:
    """Common Pile's markdown from the abstract on, as paragraphs and `##` headings, without the references."""
    rows = markdown.split("\n")
    headings = [i for i, row in enumerate(rows) if HEADING.match(row)]
    abstract = next(
        (i for i in headings if (HEADING.match(rows[i])[2] or "").lower().startswith("abstract")),
        None,
    )
    if abstract is None:
        abstract = next((i for i, row in enumerate(rows) if ABSTRACT.match(row)), None)
    # Without an abstract the text starts at the heading after the title, past the author block.
    start = abstract if abstract is not None else next(iter(headings[1:]), 1)
    out: list[str] = []
    current = ""
    for row in rows[start:]:
        found = HEADING.match(row)
        if found or not row.strip() or ITEM.match(row) or row.lstrip().startswith("|"):
            if current:
                out.append(current)
            current = ""
        if found:
            level, name = len(found[1]), found[2] or ""
            if not name:
                continue
            key = NUMBERING.sub("", name.lower()).strip(" .:")
            if key in REFERENCES:
                break
            if key.startswith("abstract"):
                out.append("## Abstract")
            else:
                out.append(f"## {name}" if level <= 4 else name)
        elif row.strip():
            current = f"{current} {row.strip()}" if current else row.strip()
    if current:
        out.append(current)
    kept = []
    for line in out:
        line = ESCAPED.sub(r"\1", line)
        if EMAIL.search(line):
            if len(line) < AFFILIATION_CHARS:
                continue
            line = " ".join(EMAIL.sub("", line).split())
        kept.append(line)
    return kept


def full_text_page(paper: Paper, body: list[str]):
    return framed(
        paper, html_url(paper.arxiv_id), heading_lines(paper) + [Line(t) for t in body]
    )


def oai_papers(path: Path) -> Iterator[tuple[str, object]]:
    for record in etree.parse(str(path)).iterfind(".//o:record", NS):
        if record.find("o:header", NS).get("status") == "deleted":
            yield (
                "deleted",
                record.findtext("o:header/o:identifier", "", NS).removeprefix(
                    "oai:arXiv.org:"
                ),
            )
            continue
        meta = record.find("o:metadata/a:arXiv", NS)
        authors = [
            squash(
                " ".join(
                    filter(
                        None,
                        (
                            a.findtext("a:forenames", "", NS),
                            a.findtext("a:keyname", "", NS),
                            a.findtext("a:suffix", "", NS),
                        ),
                    )
                )
            )
            for a in meta.iterfind("a:authors/a:author", NS)
        ]
        yield (
            "paper",
            Paper(
                meta.findtext("a:id", "", NS).strip(),
                squash(meta.findtext("a:title", "", NS)),
                authors,
                squash(meta.findtext("a:abstract", "", NS)),
                squash(meta.findtext("a:categories", "", NS)),
                meta.findtext("a:created", "", NS).strip(),
                meta.findtext("a:updated", "", NS).strip(),
                squash(meta.findtext("a:comments", "", NS)),
                squash(meta.findtext("a:journal-ref", "", NS)),
                squash(meta.findtext("a:doi", "", NS)),
                squash(meta.findtext("a:license", "", NS)),
            ),
        )


def oai_records(path: Path) -> Iterator[tuple[str, object]]:
    for kind, value in oai_papers(path):
        yield (kind, value) if kind == "deleted" else ("page", page_of(value))


def kaggle_date(text: str) -> str:
    try:
        return parsedate_to_datetime(text).strftime("%Y-%m-%d")
    except (TypeError, ValueError):
        return ""


def kaggle_records(path: Path) -> Iterator[tuple[str, object]]:
    with open(path, encoding="utf-8") as source:
        for line in source:
            item = json.loads(line)
            versions = item.get("versions") or []
            authors = [
                squash(
                    " ".join(
                        filter(
                            None,
                            (
                                p[1] if len(p) > 1 else "",
                                p[0],
                                p[2] if len(p) > 2 else "",
                            ),
                        )
                    )
                )
                for p in item.get("authors_parsed") or []
            ]
            yield (
                "page",
                page_of(
                    Paper(
                        item["id"],
                        squash(item.get("title")),
                        authors,
                        squash(item.get("abstract")),
                        squash(item.get("categories")),
                        kaggle_date(versions[0]["created"]) if versions else "",
                        kaggle_date(versions[-1]["created"]) if versions else "",
                        squash(item.get("comments")),
                        squash(item.get("journal-ref")),
                        squash(item.get("doi")),
                        squash(item.get("license")),
                    )
                ),
            )


def run(
    files: list[Path], batches: Batches, as_of: datetime, limit: int = 0
) -> Counter:
    counts: Counter = Counter()
    for path in files:
        found = (
            kaggle_records(path)
            if path.suffix in (".json", ".jsonl")
            else oai_records(path)
        )
        for kind, value in found:
            if kind == "deleted":
                batches.remove(url_of(value))
                counts["deleted"] += 1
                continue
            url, page = value
            if not page.title or len(page.text) <= len(page.title) + 1:
                counts["empty"] += 1
                continue
            batches.add(upload_row(url, page, as_of, "application/xml"))
            counts["pages"] += 1
            if limit and counts["pages"] >= limit:
                return counts
    return counts


def open_papers(paths: list[Path]) -> dict[str, Paper]:
    """The openly licensed papers among OAI-PMH pages, by arXiv id."""
    return {
        paper.arxiv_id: paper
        for path in paths
        for kind, paper in oai_papers(path)
        if kind == "paper" and is_open(paper.licence)
    }


def run_text(
    files: list[Path],
    papers: dict[str, Paper],
    batches: Batches,
    as_of: datetime,
    limit: int = 0,
) -> Counter:
    """Full-text pages from Common Pile's arXiv files, for papers our OAI records say are openly licensed."""
    counts: Counter = Counter()
    for path in files:
        with gzip.open(path, "rt", encoding="utf-8") as source:
            for line in source:
                item = json.loads(line)
                paper = papers.get(item["id"])
                if paper is None:
                    counts["not_open"] += 1
                    continue
                body = body_lines(item["text"])
                if sum(len(t) for t in body) < MIN_TEXT_CHARS:
                    counts["short"] += 1
                    continue
                url, page = full_text_page(paper, body)
                batches.add(upload_row(url, page, as_of, "text/html"))
                counts["pages"] += 1
                if limit and counts["pages"] >= limit:
                    return counts
    return counts
