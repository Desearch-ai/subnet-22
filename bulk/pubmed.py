"""PubMed baseline and update files: one page per citation with an abstract, and removals for deleted citations."""

from __future__ import annotations

import gzip
from collections import Counter
from collections.abc import Iterator
from datetime import datetime
from pathlib import Path

from lxml import etree

from bulk.rows import Batches, authors_line, paper_page, upload_row
from desearch.extraction.text.markdown import Line

SITE = "PubMed"
MONTHS = {
    m: i
    for i, m in enumerate(
        (
            "jan",
            "feb",
            "mar",
            "apr",
            "may",
            "jun",
            "jul",
            "aug",
            "sep",
            "oct",
            "nov",
            "dec",
        ),
        1,
    )
}


def url_of(pmid: str) -> str:
    return f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"


def text_of(node: etree._Element | None) -> str:
    return " ".join("".join(node.itertext()).split()) if node is not None else ""


def full_date(node: etree._Element | None) -> str:
    """YYYY-MM-DD when the node names a year, month and day, else empty."""
    if node is None:
        return ""
    year, month, day = (node.findtext(k) or "" for k in ("Year", "Month", "Day"))
    month = str(MONTHS.get(month[:3].lower(), month))
    if not (year.isdigit() and month.isdigit() and day.isdigit()):
        return ""
    return f"{int(year):04d}-{int(month):02d}-{int(day):02d}"


def author_name(author: etree._Element) -> str:
    collective = author.findtext("CollectiveName")
    if collective:
        return " ".join(collective.split())
    return " ".join(
        p for p in (author.findtext("ForeName"), author.findtext("LastName")) if p
    )


def abstract_lines(abstract: etree._Element) -> list[Line]:
    lines = []
    for part in abstract.findall("AbstractText"):
        label = (part.get("Label") or "").strip()
        if label:
            lines.append(Line(f"## {label.capitalize() if label.isupper() else label}"))
        text = text_of(part)
        if text:
            lines.append(Line(text))
    return lines


def article_page(article: etree._Element):
    """The URL and page of one PubmedArticle, or a reason it has none."""
    citation = article.find("MedlineCitation")
    pmid = citation.findtext("PMID")
    body = citation.find("Article")
    abstract = body.find("Abstract")
    if not pmid or abstract is None or not abstract.findall("AbstractText"):
        return None, "no_abstract"
    title = text_of(body.find("ArticleTitle")) or text_of(body.find("VernacularTitle"))
    if not title:
        return None, "no_title"
    journal = body.find("Journal")
    published = full_date(body.find("ArticleDate")) or full_date(
        journal.find("JournalIssue/PubDate")
    )
    issue = journal.find("JournalIssue/PubDate")
    when = (
        published
        or " ".join(t for t in (issue.findtext("Year"), issue.findtext("Month")) if t)
        or (issue.findtext("MedlineDate") or "")
    )
    doi = next(
        (
            i.text
            for i in article.iterfind("PubmedData/ArticleIdList/ArticleId")
            if i.get("IdType") == "doi" and i.text
        ),
        "",
    )
    authors = authors_line([author_name(a) for a in body.iterfind("AuthorList/Author")])
    source = ". ".join(
        p for p in (journal.findtext("Title"), when, f"doi: {doi}" if doi else "") if p
    )
    lines = [Line(t) for t in (authors, source) if t] + abstract_lines(abstract)
    languages = [l.text for l in body.iterfind("Language") if l.text]
    tags = [text_of(k) for k in citation.iterfind("KeywordList/Keyword")] + [
        text_of(d)
        for d in citation.iterfind("MeshHeadingList/MeshHeading/DescriptorName")
    ]
    url = url_of(pmid)
    page = paper_page(
        title,
        lines,
        lang="en" if "eng" in languages else (languages[0] if languages else ""),
        url=url,
        published=published,
        author=authors,
        site_name=SITE,
        tags=[t for t in tags if t],
    )
    return (url, page), None


def records(path: Path) -> Iterator[tuple[str, etree._Element]]:
    """Each PubmedArticle and DeleteCitation PMID of one file, as ("article", node) or ("deleted", pmid node)."""
    with gzip.open(path) if path.suffix == ".gz" else open(path, "rb") as source:
        for _, node in etree.iterparse(
            source, events=("end",), tag=("PubmedArticle", "DeleteCitation")
        ):
            if node.tag == "PubmedArticle":
                yield "article", node
            else:
                for pmid in node.iterfind("PMID"):
                    yield "deleted", pmid
            node.clear()
            while node.getprevious() is not None:
                del node.getparent()[0]


def run(
    files: list[Path], batches: Batches, as_of: datetime, skip: set[str], limit: int = 0
) -> Counter:
    counts: Counter = Counter()
    for path in files:
        for kind, node in records(path):
            if kind == "deleted":
                batches.remove(url_of(node.text.strip()))
                counts["deleted"] += 1
                continue
            pmid = node.findtext("MedlineCitation/PMID") or ""
            if pmid in skip:
                counts["skipped_pmid"] += 1
                continue
            found, why = article_page(node)
            if found is None:
                counts[why] += 1
                continue
            url, page = found
            batches.add(upload_row(url, page, as_of, "application/xml"))
            counts["pages"] += 1
            if limit and counts["pages"] >= limit:
                return counts
    return counts
