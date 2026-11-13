import base64
import hashlib
import importlib.util
import io
import json
import os
import struct
import subprocess
import sys
from pathlib import Path

import pytest
from lxml import etree

from desearch.extraction import worker
from tests.extraction.pages import ARTICLE, PROSE, STORY_URL

ROOT = Path(__file__).resolve().parents[2]
GOLDEN = Path(__file__).with_name("worker_golden.json")
EDGAR = "https://www.sec.gov/Archives/edgar/data/1001/000000100126000001"
CHALLENGE = (
    "<html><head><title>Just a moment...</title></head>"
    "<body><div id='cf-browser-verification'>Checking your browser before accessing the site.</div></body></html>"
)
SHORT_PAGE = "<html><head><title>GDP data</title></head><body><p>Gross domestic product, 2025: 29,184.9</p></body></html>"
SENTENCE = (
    "The company reported revenue growth across its segments during the fiscal year"
)
TEN_K = (
    "<html><head><title>acme-10k.htm</title></head><body>"
    + "".join(f"<p>{SENTENCE}, note {i}.</p>" for i in range(3000))
    + "</body></html>"
)
FILING_INDEX = (
    "<html><head><title>EDGAR Filing Documents for 0000001001-26-000001</title></head><body>"
    "<h1>Filing Detail</h1><table>"
    + "".join(
        f"<tr><td>{i}</td><td><a href='/Archives/edgar/data/1001/000000100126000001/ex{i}.htm'>Exhibit {i}</a></td>"
        f"<td>EX-{i}</td><td>{1000 + i}</td></tr>"
        for i in range(60)
    )
    + "</table></body></html>"
)
UNICODE_PAGE = (
    "<html lang='fr'><head><title>Café — 東京 🚆</title>"
    '<script type="application/ld+json">{"@type": "NewsArticle", "headline": "Caf\\u00e9 \\ud800 bad"}</script>'
    f"</head><body><article><h1>Café crème à Tōkyō</h1><p>{PROSE * 4} Ünïcödé 東京 🚆.</p></article></body></html>"
)


def escape(text: str) -> str:
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def pdf(pages: list[list[str]], title: str | None = None) -> bytes:
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    kids = []
    for lines in pages:
        ops = " ".join(f"({escape(line)}) Tj 0 -14 Td" for line in lines)
        stream = f"BT /F1 11 Tf 72 720 Td {ops} ET".encode("latin-1")
        objects.append(
            b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream"
        )
        objects.append(
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 3 0 R >> >> /Contents %d 0 R >>"
            % len(objects)
        )
        kids.append(len(objects))
    objects[1] = b"<< /Type /Pages /Kids [%s] /Count %d >>" % (
        b" ".join(b"%d 0 R" % k for k in kids),
        len(kids),
    )
    if title is not None:
        objects.append(f"<< /Title ({escape(title)}) >>".encode("latin-1"))
    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for number, body in enumerate(objects, 1):
        offsets.append(len(out))
        out += b"%d 0 obj\n" % number + body + b"\nendobj\n"
    xref = len(out)
    out += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objects) + 1)
    out += b"".join(b"%010d 00000 n \n" % offset for offset in offsets)
    info = b" /Info %d 0 R" % len(objects) if title is not None else b""
    out += b"trailer\n<< /Size %d /Root 1 0 R%s >>\nstartxref\n%d\n%%%%EOF\n" % (
        len(objects) + 1,
        info,
        xref,
    )
    return bytes(out)


def encrypted(body: bytes, user_password: str) -> bytes:
    from pypdf import PdfReader, PdfWriter

    writer = PdfWriter(clone_from=PdfReader(io.BytesIO(body)))
    writer.encrypt(
        user_password=user_password, owner_password="owner", algorithm="RC4-128"
    )
    out = io.BytesIO()
    writer.write(out)
    return out.getvalue()


def report_pages() -> list[list[str]]:
    return [
        [f"{SENTENCE} (page {n}, line {i})." for i in range(4)] for n in range(1, 4)
    ]


def document(url: str, body: bytes, content_type: str = "", which=None) -> dict:
    return {
        "url": url,
        "body_b64": base64.b64encode(body).decode(),
        "document": which,
        "content_type": content_type,
    }


def crawled(url: str, html: str, whole: bool = False, orig: str | None = None) -> dict:
    request = {"url": url, "html": html, "whole": whole}
    if orig is not None:
        request["orig"] = orig
    return request


def document_cases() -> dict:
    csv_body = b'name,city,score\n"Smith, Ann", Paris ,10\n,,\n  Bob ,"Tokyo",7\n'
    return {
        "csv": document(
            "https://data.example/files/annual_report-2025%20final.csv", csv_body
        ),
        "csv_by_content_type": document(
            "https://data.example/download?id=7", csv_body, "text/csv"
        ),
        "csv_named_by_the_record": document(
            "https://data.example/table", csv_body, "text/html", "csv"
        ),
        "csv_rows_capped": document(
            "https://data.example/big.csv",
            "".join(f"{i},{i * 2}\n" for i in range(5100)).encode(),
            "application/octet-stream",
        ),
        "csv_bad_utf8": document(
            "https://data.example/latin.csv", b"caf\xe9,cr\xe8me\n\xff,ok\n"
        ),
        "csv_only_blank_cells": document(
            "https://data.example/blank.csv", b" , ,\n,,\n"
        ),
        "csv_too_large": document("https://data.example/huge.csv", b"a,b\n" * 50_001),
        "tsv": document(
            "https://data.example/t.tsv",
            b"a\tb\tc\n1\t\t3\n",
            "application/octet-stream",
        ),
        "json": document(
            "https://data.example/x.json",
            json.dumps(
                {"name": "Zürich", "values": [1, 2.5, None], "nested": {"ok": True}}
            ).encode(),
            "application/json",
        ),
        "json_lone_surrogate": document(
            "https://data.example/s.json", b'{"text": "a\\ud800b"}', "application/json"
        ),
        "json_unreadable": document(
            "https://data.example/bad.json", b"{not json", "application/json"
        ),
        "jsonl": document(
            "https://data.example/rows.jsonl", b'{"a": 1}\n\n{"b": [2, 3]}\n'
        ),
        "txt": document(
            "https://data.example/notes.txt",
            "  First line  \n\n\tSecond line\nThird — line\n".encode(),
            "text/plain",
        ),
        "txt_long": document(
            "https://data.example/long.txt",
            ("x" * 99 + "\n").encode() * 1990,
            "text/plain",
        ),
        "txt_whitespace_only": document(
            "https://data.example/empty.txt", b"  \n\t\n", "text/plain"
        ),
        "xls": document(
            "https://data.example/sheet.xls",
            b"\xd0\xcf\x11\xe0",
            "application/vnd.ms-excel",
        ),
        "xlsx_by_extension": document(
            "https://data.example/sheet.xlsx", b"PK\x03\x04", "application/octet-stream"
        ),
        "html_is_not_a_document": document(
            "https://data.example/page.html", b"<html></html>", "text/html"
        ),
        "unknown_extension": document("https://data.example/file.bin", b"\x00\x01", ""),
        "host_named_file": document("https://data.example/", b"a,b\n1,2\n", "", "csv"),
    }


def pdf_cases() -> dict:
    good = pdf(report_pages(), "Acme Annual Report 2025")
    return {
        "pdf_meta_title": document(
            "https://acme.example/reports/annual.pdf", good, "application/pdf"
        ),
        "pdf_poor_title": document(
            "https://acme.example/r.pdf",
            pdf(report_pages(), "Microsoft Word - draft.docx"),
            "application/pdf",
        ),
        "pdf_no_title": document(
            "https://acme.example/r2.pdf", pdf(report_pages()), "application/pdf"
        ),
        "pdf_by_magic_bytes": document(
            "https://acme.example/view?doc=1", good, "text/html"
        ),
        "pdf_short": document(
            "https://acme.example/s.pdf", pdf([["Too short."]]), "application/pdf"
        ),
        "pdf_unreadable": document(
            "https://acme.example/bad.pdf",
            b"%PDF-1.4\nnot a pdf at all",
            "application/pdf",
        ),
        "pdf_encrypted": document(
            "https://acme.example/locked.pdf",
            encrypted(good, "secret"),
            "application/pdf",
        ),
        "pdf_owner_password_only": document(
            "https://acme.example/open.pdf", encrypted(good, ""), "application/pdf"
        ),
        "pdf_pages_capped": document(
            "https://acme.example/many.pdf",
            pdf([[f"{SENTENCE} on page {n}."] for n in range(1, 403)], "Many Pages"),
            "application/pdf",
        ),
        "pdf_blank_first_page": document(
            "https://acme.example/blank-first.pdf",
            pdf([[], *report_pages()], "Untitled"),
            "application/pdf",
        ),
    }


def crawled_cases() -> dict:
    return {
        "article": crawled(STORY_URL, ARTICLE),
        "challenge": crawled("https://shop.example/item/7", CHALLENGE),
        "long_page_fetched_as_a_challenge": crawled(STORY_URL, ARTICLE, orig=CHALLENGE),
        "short_page": crawled("https://fred.example/data/GDP", SHORT_PAGE),
        "short_page_fetched_as_a_challenge": crawled(
            "https://fred.example/data/GDP", SHORT_PAGE, orig=CHALLENGE
        ),
        "unicode_and_surrogates": crawled("https://café.example/tokyo", UNICODE_PAGE),
        "edgar_whole": crawled(f"{EDGAR}/acme-10k.htm", TEN_K, whole=True),
        "edgar_not_whole": crawled(f"{EDGAR}/acme-10k.htm", TEN_K),
        "edgar_article_not_whole": crawled(f"{EDGAR}/letter.htm", ARTICLE),
        "edgar_listing": crawled(
            f"{EDGAR}/0000001001-26-000001-index.htm", FILING_INDEX
        ),
        "edgar_listing_whole": crawled(
            f"{EDGAR}/0000001001-26-000001-index.htm", FILING_INDEX, whole=True
        ),
        "edgar_r_file": crawled(f"{EDGAR}/R2.htm", TEN_K, whole=True),
        "edgar_text_submission": crawled(
            f"{EDGAR}/0000001001-26-000001.txt", "\n".join([SENTENCE] * 50), whole=True
        ),
        "edgar_empty": crawled(f"{EDGAR}/empty.htm", "", whole=True),
        "not_edgar_whole": crawled(
            "https://www.sec.gov/news/press-release/2026-1", TEN_K, whole=True
        ),
    }


def cases(with_pdfs: bool) -> dict:
    found = {name: (worker.DOCUMENT, r) for name, r in document_cases().items()}
    if with_pdfs:
        found.update({name: (worker.DOCUMENT, r) for name, r in pdf_cases().items()})
    found.update({name: (worker.CRAWLED, r) for name, r in crawled_cases().items()})
    return found


def digest(answer: dict) -> dict:
    """The answer with long page texts as their hash and length."""
    page = answer.get("page")
    if page and len(page["text"]) > 2000:
        text = page["text"]
        page = {
            **page,
            "text": {
                "sha256": hashlib.sha256(text.encode()).hexdigest(),
                "chars": len(text),
            },
        }
        return {**answer, "page": page}
    return answer


def test_documents_and_crawled_pages_answer_as_the_golden_answers():
    golden = json.loads(GOLDEN.read_text())
    with_pdfs = importlib.util.find_spec("pypdf") is not None
    found = cases(with_pdfs)
    assert set(found) == {n for n in golden if with_pdfs or not n.startswith("pdf_")}
    for name, (op, request) in found.items():
        got = json.loads(worker.reply(op + json.dumps(request).encode()))
        assert digest(got) == golden[name], name


def frame(op: bytes, data: bytes) -> bytes:
    return struct.pack(">I", len(op) + len(data)) + op + data


def answers(stdout: bytes) -> list[dict]:
    found, at = [], 0
    while at < len(stdout):
        (size,) = struct.unpack(">I", stdout[at : at + 4])
        found.append(json.loads(stdout[at + 4 : at + 4 + size]))
        at += 4 + size
    return found


@pytest.mark.parametrize("launch", ["module", "source"])
def test_the_worker_answers_framed_requests_run_as_a_module_or_from_its_source(
    tmp_path, launch
):
    url = STORY_URL.encode()
    requests = [
        frame(worker.EXTRACT, struct.pack(">I", len(url)) + url + ARTICLE.encode()),
        frame(
            worker.CRAWLED,
            json.dumps(
                crawled("https://fred.example/data/GDP", SHORT_PAGE, orig=CHALLENGE)
            ).encode(),
        ),
        frame(
            worker.DOCUMENT,
            json.dumps(document("https://d.example/t.csv", b"a,b\n1,2\n")).encode(),
        ),
        frame(worker.BODY, b"<p>One <a href='/x'>link</a></p>"),
        frame(worker.ANCHORS, b"<a href='/one'>1</a><a>2</a>"),
        frame(
            worker.PMC,
            b"<article><front><article-meta><abstract><p>Short.</p></abstract></article-meta></front></article>",
        ),
        frame(worker.PMC, b"<article"),
        frame(b"z", b""),
    ]
    program = (
        ["-m", "desearch.extraction.worker"]
        if launch == "module"
        else ["-c", Path(worker.__file__).read_text()]
    )
    done = subprocess.run(
        [sys.executable, *program],
        input=b"".join(requests),
        capture_output=True,
        cwd=tmp_path,
        env={**os.environ, "PYTHONPATH": str(ROOT)},
        timeout=120,
        check=True,
    )
    got = answers(done.stdout)
    assert got[:7] == [json.loads(worker.reply(r[4:])) for r in requests[:7]]
    assert got[0]["page"]["title"] == "Council approves transit plan"
    assert (got[1]["blocked"], got[2]["which"]) == (True, "csv")
    assert got[4] == {"hrefs": ["/one", None]}
    assert (got[6], got[7]["error"]) == ({"error": "xml"}, "crash")


def article(revision: int, body: str, **extra) -> dict:
    html = (
        '<!DOCTYPE html><html><head><meta property="mw:pageNamespace" content="0"/>'
        '<link rel="mw:PageProp/Category" href="./Category:Squirrels"/><link rel="mw:PageProp/Category" href="./Category:Articles_with_short_description"/>'
        f"<title>Squirrel</title></head><body>{body}</body></html>"
    )
    return {
        "name": "Squirrel",
        "identifier": 28492,
        "url": "https://en.wikipedia.org/wiki/Squirrel",
        "namespace": {"identifier": 0},
        "in_language": {"identifier": "en"},
        "is_part_of": {"identifier": "enwiki"},
        "date_created": "2001-10-09T17:55:09Z",
        "date_modified": "2026-10-26T10:28:11Z",
        "version": {"identifier": revision},
        "article_body": {"html": html},
        **extra,
    }


SQUIRREL = '<section data-mw-section-id="0"><p>Squirrels are rodents of the family Sciuridae.</p></section><section><h2 id="Tail">Tail</h2><p>The tail is bushy.</p></section>'


def test_a_snapshot_article_is_framed_like_the_live_page():
    framed, disambiguation = worker.page_html(
        article(1, SQUIRREL, main_entity={"identifier": "Q9482"}),
        {"Articles with short description"},
    )
    assert not disambiguation
    root = etree.HTML(framed)
    assert root.findtext(".//title") == "Squirrel - Wikipedia"
    assert [a.text for a in root.iterfind(".//*[@id='mw-normal-catlinks']//a")] == [
        "Squirrels"
    ]
    ld = json.loads(root.findtext(".//script"))
    assert (ld["sameAs"], ld["datePublished"], ld["dateModified"]) == (
        "http://www.wikidata.org/entity/Q9482",
        "2001-10-09T17:55:09Z",
        "2026-10-26T10:28:11Z",
    )


def test_the_page_image_follows_mediawikis_og_image_rule():
    host = "en.wikipedia.org"
    small = {
        "content_url": "https://upload.wikimedia.org/wikipedia/commons/6/68/Sciuridae.jpg",
        "width": 600,
        "height": 600,
    }
    tall = {
        "content_url": "https://upload.wikimedia.org/wikipedia/commons/4/43/Sir_Robert.jpg",
        "width": 1022,
        "height": 1500,
    }
    vector = {
        "content_url": "https://upload.wikimedia.org/wikipedia/commons/1/18/ISO_C%2B%2B_Logo.svg",
        "width": 306,
        "height": 344,
    }
    assert worker.og_image(small, host).endswith(
        "Sciuridae.jpg?utm_source=en.wikipedia.org&utm_campaign=index&utm_content=thumbnail_unscaled"
    )
    assert worker.og_image(tall, host).startswith(
        "https://thumb.wikimedia.org/wikipedia/commons/thumb/4/43/Sir_Robert.jpg/960px-Sir_Robert.jpg?"
    )
    assert (
        "/thumb/1/18/ISO_C%2B%2B_Logo.svg/1280px-ISO_C%2B%2B_Logo.svg.png?"
        in worker.og_image(vector, host)
    )


def test_disambiguation_pages_are_marked():
    disambiguation = article(9, "<p>Squirrel may refer to many things.</p>")
    disambiguation["article_body"]["html"] = disambiguation["article_body"][
        "html"
    ].replace("<title>", '<meta property="mw:PageProp/disambiguation"/><title>')
    assert worker.page_html(disambiguation, set())[1]
    assert not worker.page_html(article(9, SQUIRREL), set())[1]
