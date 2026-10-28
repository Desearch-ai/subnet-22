"""arXiv abstracts (CC0 metadata): one page per paper, from OAI-PMH pages or the Kaggle metadata snapshot."""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from collections.abc import Iterator
from datetime import datetime
from email.utils import parsedate_to_datetime
from pathlib import Path

from lxml import etree

from bulk.rows import Batches, authors_line, paper_page, upload_row
from desearch.extraction.text.markdown import Line

OAI = "https://oaipmh.arxiv.org/oai"
NS = {"o": "http://www.openarchives.org/OAI/2.0/", "a": "http://arxiv.org/OAI/arXiv/"}
# arXiv's robots.txt asks for 15 s between requests; OAI-PMH asks for 3 s and sends Retry-After when busy.
DELAY_S = 15.0
SITE = "arXiv"


def url_of(arxiv_id: str) -> str:
    return f"https://arxiv.org/abs/{arxiv_id}"


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


def page_of(
    arxiv_id: str,
    title: str,
    authors: list[str],
    abstract: str,
    categories: str,
    created: str,
    updated: str,
    comments: str,
    journal: str,
    doi: str,
):
    stamp = f"Submitted {created}" + (
        f", updated {updated}" if updated and updated != created else ""
    )
    lines = [
        Line(authors_line(authors)),
        Line(" · ".join(p for p in (f"arXiv:{arxiv_id}", categories, stamp) if p)),
        Line("## Abstract"),
        Line(abstract),
    ]
    lines += [
        Line(f"{name}: {value}")
        for name, value in (
            ("Comments", comments),
            ("Journal reference", journal),
            ("DOI", doi),
        )
        if value
    ]
    url = url_of(arxiv_id)
    return url, paper_page(
        title,
        lines,
        lang="en",
        url=url,
        published=created,
        modified=updated if updated != created else "",
        author=authors_line(authors),
        site_name=SITE,
        tags=categories.split(),
    )


def oai_records(path: Path) -> Iterator[tuple[str, object]]:
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
            "page",
            page_of(
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
            ),
        )


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
