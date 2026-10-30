"""PMC open-access full text under commercial-use licenses, from the anonymous `pmc-oa-opendata` bucket."""

from __future__ import annotations

import csv
import gzip
import json
import re
import urllib.request
from collections import Counter
from collections.abc import Iterable, Iterator
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

from lxml import etree
from lxml import html as lhtml

from bulk.rows import Batches, authors_line, paper_page, upload_row
from desearch.extraction.text.markdown import Line, render

BUCKET = "https://pmc-oa-opendata.s3.amazonaws.com"
SEPARATOR = re.compile(r"[\s,;\u2013\u2014-]*")
EMPTY_BRACKETS = re.compile(r"\s*(\[[\s,;\u2013-]*\]|\([\s,;\u2013-]*\))")
SITE = "PMC"
COMMERCIAL = frozenset(("CC0", "CC BY", "CC BY-SA", "CC BY-ND"))
DROPPED = (
    "label",
    "disp-formula",
    "inline-formula",
    "supplementary-material",
    "fn-group",
    "alternatives",
    "ref-list",
    "graphic",
    "inline-graphic",
    "media",
    "table-wrap-foot",
    "object-id",
)
BLOCK_TAGS = {
    "sec": "div",
    "p": "p",
    "list": "ul",
    "list-item": "li",
    "table-wrap": "div",
    "fig": "div",
    "caption": "p",
    "boxed-text": "div",
    "statement": "div",
    "def-list": "ul",
    "def-item": "li",
    "disp-quote": "blockquote",
    "verse-group": "div",
    "preformat": "pre",
    "app": "div",
    "app-group": "div",
}
INLINE_TAGS = {"italic": "i", "bold": "b", "sup": "sup", "sub": "sub", "term": "b"}


def url_of(pmcid: str) -> str:
    return f"https://pmc.ncbi.nlm.nih.gov/articles/{pmcid}/"


def text_of(node: etree._Element | None) -> str:
    return " ".join("".join(node.itertext()).split()) if node is not None else ""


def local_name(node: etree._Element) -> str:
    return etree.QName(node).localname if isinstance(node.tag, str) else ""


def as_html(node: etree._Element, depth: int = 2) -> etree._Element:
    """A JATS subtree with HTML tags render() reads: sections as divs led by `h2`-`h4` titles, citation marks and formulas dropped."""
    tree = etree.fromstring(etree.tostring(node))
    drop_citations(tree)
    for drop in [e for e in tree.iter() if dropped(e, tree)]:
        if drop.getparent() is not None:
            drop_keep_tail(drop)
    tags = [(e, html_tag(e, tree, depth)) for e in tree.iter() if isinstance(e.tag, str)]
    for element, tag in tags:
        element.tag = tag
        element.attrib.clear()
    etree.cleanup_namespaces(tree)
    return lhtml.fromstring(etree.tostring(tree, encoding="unicode"))


def is_citation(element: etree._Element | None) -> bool:
    return element is not None and local_name(element) == "xref" and element.get("ref-type") == "bibr"


def drop_citations(tree: etree._Element) -> None:
    """Citation marks out, with the commas and dashes between them, then any brackets or superscripts left empty."""
    for xref in [e for e in tree.iter() if is_citation(e)]:
        if is_citation(xref.getnext()) and SEPARATOR.fullmatch(xref.tail or ""):
            xref.tail = ""
        drop_keep_tail(xref)
    for mark in [e for e in tree.iter() if local_name(e) in ("sup", "sub") and len(e) == 0 and SEPARATOR.fullmatch(e.text or "")]:
        drop_keep_tail(mark)
    for element in tree.iter():
        if isinstance(element.tag, str):
            element.text = element.text and EMPTY_BRACKETS.sub("", element.text)
            element.tail = element.tail and EMPTY_BRACKETS.sub("", element.tail)


def dropped(element: etree._Element, tree: etree._Element) -> bool:
    name = local_name(element)
    if name == "title" and element.getparent() is tree and local_name(tree) == "abstract":
        return True
    return name in DROPPED


def html_tag(element: etree._Element, tree: etree._Element, depth: int) -> str:
    name = local_name(element)
    if element is tree:
        return "div"
    if name == "title" and local_name(element.getparent()) in ("sec", "app", "ack", "boxed-text"):
        sections = sum(1 for a in element.iterancestors() if local_name(a) == "sec")
        return f"h{max(depth, min(4, depth + sections - 1))}"
    if name == "list" and element.get("list-type") == "order":
        return "ol"
    if name in BLOCK_TAGS:
        return BLOCK_TAGS[name]
    if name in INLINE_TAGS:
        return INLINE_TAGS[name]
    if name in ("table", "thead", "tbody", "tr", "th", "td"):
        return name
    return "span"


def drop_keep_tail(node: etree._Element) -> None:
    parent, tail = node.getparent(), node.tail or ""
    previous = node.getprevious()
    if previous is not None:
        previous.tail = (previous.tail or "") + tail
    else:
        parent.text = (parent.text or "") + tail
    parent.remove(node)


def full_date(node: etree._Element | None) -> str:
    if node is None:
        return ""
    year, month, day = (node.findtext(k) or "" for k in ("year", "month", "day"))
    if not (year.isdigit() and month.isdigit() and day.isdigit()):
        return ""
    return f"{int(year):04d}-{int(month):02d}-{int(day):02d}"


def article_page(xml: bytes, meta: dict):
    """The URL and page of one article version, or a reason it has none."""
    root = etree.fromstring(xml)
    front = root.find("front/article-meta")
    if front is None:
        return None, "no_front"
    title = text_of(front.find("title-group/article-title"))
    if not title:
        return None, "no_title"
    authors = []
    for contrib in front.iterfind("contrib-group/contrib"):
        if contrib.get("contrib-type", "author") != "author":
            continue
        name = contrib.find("name")
        if name is not None:
            authors.append(
                " ".join(
                    p
                    for p in (
                        text_of(name.find("given-names")),
                        text_of(name.find("surname")),
                    )
                    if p
                )
            )
        elif contrib.find("collab") is not None:
            authors.append(text_of(contrib.find("collab")))
    lines = [
        Line(t)
        for t in (authors_line(authors), " ".join((meta.get("citation") or "").split()))
        if t
    ]
    for abstract in front.iterfind("abstract"):
        if abstract.get("abstract-type") not in (None, "summary", "structured"):
            continue
        lines.append(Line("## Abstract"))
        lines += render(as_html(abstract, depth=3))
    body = root.find("body")
    if body is not None:
        lines += render(as_html(body))
    dates = {
        d.get("pub-type") or d.get("date-type"): d for d in front.iterfind("pub-date")
    }
    published = next(
        (
            full_date(dates[k])
            for k in ("epub", "pub", "ppub", "collection")
            if k in dates and full_date(dates[k])
        ),
        "",
    )
    lang = root.get("{http://www.w3.org/XML/1998/namespace}lang") or "en"
    url = url_of(meta["pmcid"])
    page = paper_page(
        title,
        [line for line in lines if line.text.strip()],
        lang=lang.lower()[:2],
        url=url,
        published=published,
        author=authors_line(authors),
        site_name=SITE,
        tags=[text_of(k) for k in front.iterfind("kwd-group/kwd")],
    )
    if body is None:
        return (url, page), "no_body"
    return (url, page), None


def left_out(meta: dict) -> str | None:
    """Why an article version stays out, or None for one we take."""
    if not meta.get("is_pmc_openaccess"):
        return "not_open_access"
    if meta.get("license_code") not in COMMERCIAL:
        return "license"
    if meta.get("is_retracted"):
        return "retracted"
    return None


def latest_versions(keys: Iterable[str]) -> dict[str, str]:
    """The newest version of each PMCID from metadata keys like `metadata/PMC13901.1.json`."""
    newest: dict[str, tuple[int, str]] = {}
    for key in keys:
        name = key.rsplit("/", 1)[-1].removesuffix(".json")
        pmcid, _, version = name.partition(".")
        if not pmcid.startswith("PMC") or not version.isdigit():
            continue
        if pmcid not in newest or int(version) > newest[pmcid][0]:
            newest[pmcid] = (int(version), name)
    return {pmcid: name for pmcid, (_, name) in newest.items()}


def inventory_keys(paths: Iterable[Path]) -> Iterator[str]:
    """Metadata keys from the bucket's S3 inventory CSV files (gzipped or not)."""
    for path in paths:
        with gzip.open(path, "rt") if path.suffix == ".gz" else open(path) as source:
            for row in csv.reader(source):
                if len(row) > 1 and row[1].startswith("metadata/"):
                    yield row[1]


def fetch(key: str, user_agent: str) -> bytes:
    request = urllib.request.Request(
        f"{BUCKET}/{key}", headers={"User-Agent": user_agent}
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        return response.read()


def from_bucket(
    names: Iterable[str], user_agent: str, workers: int
) -> Iterator[tuple[dict, bytes | None]]:
    """Each version's metadata, with its XML only when the version is taken."""

    def one(name: str) -> tuple[dict, bytes | None]:
        meta = json.loads(fetch(f"metadata/{name}.json", user_agent))
        return meta, None if left_out(meta) else fetch(f"{name}/{name}.xml", user_agent)

    with ThreadPoolExecutor(workers) as pool:
        yield from pool.map(one, names)


def taken_pmids(names: Iterable[str], user_agent: str, workers: int) -> Iterator[str]:
    """PMIDs of the versions the PMC run takes, from their metadata alone, so PubMed can leave those abstracts out first."""

    def one(name: str) -> str:
        meta = json.loads(fetch(f"metadata/{name}.json", user_agent))
        return "" if left_out(meta) else str(meta.get("pmid") or "")

    with ThreadPoolExecutor(workers) as pool:
        yield from (pmid for pmid in pool.map(one, names) if pmid)


def from_folder(folder: Path) -> Iterator[tuple[dict, bytes | None]]:
    """`<name>.json` and `<name>.xml` pairs in one folder, for tests and replays."""
    for path in sorted(folder.glob("PMC*.json")):
        meta = json.loads(path.read_text())
        xml = path.with_suffix(".xml")
        yield meta, xml.read_bytes() if xml.exists() and not left_out(meta) else None


def run(
    articles: Iterable[tuple[dict, bytes | None]],
    batches: Batches,
    as_of: datetime,
    limit: int = 0,
    pmids: list[str] | None = None,
) -> Counter:
    """Pages of the versions taken; `pmids` collects their PMIDs so the PubMed run can leave those abstracts out."""
    counts: Counter = Counter()
    for meta, xml in articles:
        why = left_out(meta)
        if why:
            counts[why] += 1
            continue
        if xml is None:
            counts["no_xml"] += 1
            continue
        try:
            found, why = article_page(xml, meta)
        except etree.XMLSyntaxError:
            counts["bad_xml"] += 1
            continue
        if found is None:
            counts[why] += 1
            continue
        if why:
            counts[why] += 1
        url, page = found
        batches.add(upload_row(url, page, as_of, "application/xml"))
        counts["pages"] += 1
        if pmids is not None and meta.get("pmid"):
            pmids.append(str(meta["pmid"]))
        if limit and counts["pages"] >= limit:
            break
    return counts


def latest_inventory(dest: Path, user_agent: str) -> list[Path]:
    """Downloads the newest daily inventory's CSV files into `dest` and returns their paths."""
    listing = fetch("?list-type=2&delimiter=/&prefix=inventory-reports/pmc-oa-opendata/metadata/", user_agent)
    days = sorted(p.text for p in etree.fromstring(listing).iter("{*}Prefix") if p.text and p.text.rstrip("/").endswith("Z"))
    manifest = json.loads(fetch(f"{days[-1]}manifest.json", user_agent))
    dest.mkdir(parents=True, exist_ok=True)
    paths = []
    for file in manifest["files"]:
        path = dest / file["key"].rsplit("/", 1)[-1]
        if not path.exists():
            path.write_bytes(fetch(file["key"], user_agent))
        paths.append(path)
    return paths
