import gzip
import json
from datetime import datetime, timezone

import pyarrow.parquet as pq
from lxml import etree

from bulk import arxiv, pmc, pubmed, wikipedia
from bulk.rows import MANIFEST, Batches
from desearch.extraction.schema import PAGE_SCHEMA

AS_OF = datetime(2026, 11, 1, tzinfo=timezone.utc)

PUBMED = """<PubmedArticleSet><PubmedArticle><MedlineCitation><PMID Version="1">41605288</PMID>
<Article><Journal><JournalIssue><PubDate><Year>2026</Year><Month>Jan</Month></PubDate></JournalIssue>
<Title>Journal of advanced research</Title></Journal>
<ArticleTitle>High-protein diet promotes <i>aging</i>.</ArticleTitle>
<Abstract><AbstractText Label="INTRODUCTION">Dietary protein has functions.</AbstractText>
<AbstractText Label="RESULTS">Lifespan fell.</AbstractText><CopyrightInformation>Copyright 2026.</CopyrightInformation></Abstract>
<AuthorList><Author><LastName>Xu</LastName><ForeName>Xiaoqing</ForeName></Author><Author><CollectiveName>The Study Group</CollectiveName></Author></AuthorList>
<Language>eng</Language><ArticleDate DateType="Electronic"><Year>2026</Year><Month>01</Month><Day>26</Day></ArticleDate></Article>
<MeshHeadingList><MeshHeading><DescriptorName>Aging</DescriptorName></MeshHeading></MeshHeadingList></MedlineCitation>
<PubmedData><ArticleIdList><ArticleId IdType="doi">10.1016/j.jare.2026.01.063</ArticleId></ArticleIdList></PubmedData></PubmedArticle>
<PubmedArticle><MedlineCitation><PMID>2</PMID><Article><Journal><Title>J</Title><JournalIssue><PubDate><Year>2020</Year></PubDate></JournalIssue></Journal>
<ArticleTitle>No abstract</ArticleTitle></Article></MedlineCitation></PubmedArticle>
<DeleteCitation><PMID Version="1">22213645</PMID></DeleteCitation></PubmedArticleSet>"""

JATS = """<article xml:lang="en"><front><article-meta><article-id pub-id-type="doi">10.1002/npr2.12310</article-id><title-group><article-title>Risk factors</article-title></title-group>
<contrib-group><contrib contrib-type="author"><name><surname>Masataka</surname><given-names>Yuji</given-names></name></contrib></contrib-group>
<pub-date pub-type="epub"><day>19</day><month>12</month><year>2022</year></pub-date>
<abstract><title>Abstract</title><sec><title>Aim</title><p>To determine risk factors.</p></sec></abstract>
<kwd-group><kwd>cannabis</kwd></kwd-group></article-meta></front>
<body><sec><title>INTRODUCTION</title><p>Laws changed in 2022.<xref ref-type="bibr" rid="b1">1</xref>, <xref ref-type="bibr" rid="b2">2</xref></p>
<p>Seen before [<xref ref-type="bibr" rid="b3">3</xref>] and after<sup><xref ref-type="bibr" rid="b4">4</xref></sup>.</p>
<sec><title>Survey</title><p>See <xref ref-type="fig" rid="f1">Figure 1</xref>.</p><fig><label>Figure 1</label><caption><p>Respondents by age.</p></caption></fig></sec></sec>
<sec><title>Methods</title><disp-formula><mml:math xmlns:mml="http://www.w3.org/1998/Math/MathML"><mml:mi>x</mml:mi></mml:math></disp-formula><p>We asked users.</p></sec></body>
<back><ref-list><ref id="b1"><mixed-citation>Someone 2020.</mixed-citation></ref></ref-list></back></article>"""

ARXIV = """<OAI-PMH xmlns="http://www.openarchives.org/OAI/2.0/"><ListRecords>
<record><header><identifier>oai:arXiv.org:1303.7276</identifier></header><metadata>
<arXiv xmlns="http://arxiv.org/OAI/arXiv/"><id>1303.7276</id><created>2013-03-29</created><updated>2026-10-26</updated>
<authors><author><keyname>Li</keyname><forenames>Chen</forenames></author></authors>
<title>The effect of limb
 kinematics</title><categories>physics.bio-ph cs.RO</categories><abstract>  Legged robots on sand.  </abstract></arXiv></metadata></record>
<record><header status="deleted"><identifier>oai:arXiv.org:hep-th/9901001</identifier></header></record>
</ListRecords></OAI-PMH>"""


def rows_of(out):
    manifest = json.loads((out / MANIFEST).read_text())
    return [
        r for name in manifest["files"] for r in pq.read_table(out / name).to_pylist()
    ], manifest


def test_pubmed_citations_become_pages_and_deleted_ones_removals(tmp_path):
    source = tmp_path / "pubmed26n0001.xml"
    source.write_text(PUBMED)
    batches = Batches(tmp_path / "run", "pubmed", "20261101", AS_OF)
    counts = pubmed.run([source], batches, AS_OF, set())
    batches.close(dict(counts))
    rows, manifest = rows_of(tmp_path / "run")
    assert (counts["pages"], counts["no_abstract"], counts["deleted"]) == (1, 1, 1)
    row = rows[0]
    assert row["url"] == "https://pubmed.ncbi.nlm.nih.gov/41605288/"
    assert (
        row["fetched_at"] == AS_OF and row["extraction"] == "v2" and row["lang"] == "en"
    )
    assert row["text"].splitlines() == [
        "High-protein diet promotes aging.",
        "Xiaoqing Xu, The Study Group",
        "Journal of advanced research. 2026-01-26. doi: 10.1016/j.jare.2026.01.063",
        "## Introduction",
        "Dietary protein has functions.",
        "## Results",
        "Lifespan fell.",
    ]
    assert row["headings"] == ["Introduction", "Results"] and row["tags"] == ["Aging"]
    assert (
        tmp_path / "run" / manifest["removed"]
    ).read_text() == "https://pubmed.ncbi.nlm.nih.gov/22213645/\n"


def test_pmc_full_text_drops_citation_marks_formulas_and_references():
    meta = {
        "pmcid": "PMC10009416",
        "citation": "Neuropsychopharmacol Rep. 2022;43(1):85-94.",
        "is_pmc_openaccess": True,
        "license_code": "CC BY",
    }
    (url, page), why = pmc.article_page(JATS.encode(), meta)
    assert why is None and url == "https://pmc.ncbi.nlm.nih.gov/articles/PMC10009416/"
    assert page.text.splitlines() == [
        "Risk factors",
        "Yuji Masataka",
        "Neuropsychopharmacol Rep. 2022;43(1):85-94.",
        "DOI: 10.1002/npr2.12310",
        "## Abstract",
        "### Aim",
        "To determine risk factors.",
        "## INTRODUCTION",
        "Laws changed in 2022.",
        "Seen before and after.",
        "### Survey",
        "See Figure 1.",
        "Respondents by age.",
        "## Methods",
        "We asked users.",
    ]
    assert (page.published, page.tags) == ("2022-12-19", ["cannabis"])


def test_pmc_takes_only_open_commercial_unretracted_versions_and_the_newest_of_each():
    assert pmc.left_out({"is_pmc_openaccess": True, "license_code": "CC BY-SA"}) is None
    assert (
        pmc.left_out({"is_pmc_openaccess": True, "license_code": "CC BY-NC"})
        == "license"
    )
    assert pmc.left_out({"is_pmc_openaccess": True, "license_code": None}) == "license"
    assert (
        pmc.left_out({"is_pmc_openaccess": False, "license_code": "CC BY"})
        == "not_open_access"
    )
    assert (
        pmc.left_out(
            {"is_pmc_openaccess": True, "license_code": "CC0", "is_retracted": True}
        )
        == "retracted"
    )
    keys = [
        "metadata/PMC1.1.json",
        "metadata/PMC1.3.json",
        "metadata/PMC1.2.json",
        "metadata/PMC2.1.json",
        "README.txt",
    ]
    assert pmc.latest_versions(keys) == {"PMC1": "PMC1.3", "PMC2": "PMC2.1"}


def test_pmc_bucket_reads_retry_server_errors_and_skip_versions_gone_since_the_inventory(monkeypatch):
    import io
    import urllib.error

    calls = {}

    def urlopen(request, timeout):
        url = request.full_url
        calls[url] = calls.get(url, 0) + 1
        if "PMC2.1" in url:
            raise urllib.error.HTTPError(url, 404, "Not Found", {}, None)
        if url.endswith("PMC1.1.json") and calls[url] < 3:
            raise urllib.error.HTTPError(url, 500, "Internal Server Error", {}, None)
        body = b'{"pmcid": "PMC1", "is_pmc_openaccess": true, "license_code": "CC BY"}' if url.endswith(".json") else b"<article/>"
        return io.BytesIO(body)

    monkeypatch.setattr(pmc.urllib.request, "urlopen", urlopen)
    monkeypatch.setattr(pmc.time, "sleep", lambda s: None)
    found = list(pmc.from_bucket(["PMC1.1", "PMC2.1"], "test", workers=1))
    assert found[0] == ({"pmcid": "PMC1", "is_pmc_openaccess": True, "license_code": "CC BY"}, b"<article/>")
    assert calls[f"{pmc.BUCKET}/metadata/PMC1.1.json"] == 3, "two server errors, then the read"
    assert pmc.left_out(found[1][0]) == "gone" and found[1][1] is None


def test_pmc_versions_go_most_cited_per_year_first_by_their_pmid_in_openalex(tmp_path):
    import pyarrow as pa
    import pyarrow.fs as pafs
    import pyarrow.parquet as pq

    ids = tmp_path / "PMC-ids.csv.gz"
    with gzip.open(ids, "wt", newline="") as out:
        out.write("Journal Title,ISSN,eISSN,Year,Volume,Issue,Page,DOI,PMCID,PMID,Manuscript Id,Release Date\n")
        out.write("J,,,2006,1,1,1,10.1/a,PMC1,11,,live\n")
        out.write("J,,,2025,1,1,1,10.1/b,PMC2,22,,live\n")
        out.write("J,,,2025,1,1,1,10.1/c,PMC3,33,,live\n")
        out.write("J,,,2024,1,1,1,10.1/d,PMC4,,,live\n")
        out.write("J,,,2020,1,1,1,10.1/e,PMC9,99,,live\n")
    works = tmp_path / "part_0000.parquet"
    pubmed = "https://pubmed.ncbi.nlm.nih.gov/"
    pq.write_table(
        pa.table(
            {
                "ids": pa.array(
                    [
                        [("openalex", "W1"), ("pmid", pubmed + "11")],
                        [("openalex", "W2"), ("pmid", pubmed + "22")],
                        [("openalex", "W3"), ("doi", "https://doi.org/10.1/c")],
                        [("openalex", "W9"), ("pmid", pubmed + "99")],
                    ],
                    type=pa.map_(pa.string(), pa.string()),
                ),
                "cited_by_count": [210, 18, 500, 70],
            }
        ),
        works,
    )
    pmids = pmc.pmc_ids(ids)
    assert pmids["PMC2"] == ("22", 2025) and pmids["PMC4"] == ("", 2024)
    cited = pmc.citations([str(works)], pafs.LocalFileSystem(), {"11", "22", "33"}, workers=1)
    assert cited == {"11": 210, "22": 18}, "PMIDs outside the wanted set are not kept"
    names = {"PMC1": "PMC1.2", "PMC2": "PMC2.1", "PMC3": "PMC3.1", "PMC4": "PMC4.1", "PMC5": "PMC5.1"}
    assert pmc.order(names, pmids, cited, 2026) == ["PMC1.2", "PMC2.1", "PMC5.1", "PMC4.1", "PMC3.1"], (
        "210 citations in 21 years beat 18 in 2; the rest go newest first"
    )


def test_arxiv_records_become_abstract_pages_and_deleted_records_removals(tmp_path):
    source = tmp_path / "oai-000001.xml"
    source.write_text(ARXIV)
    batches = Batches(tmp_path / "run", "arxiv", "20261101", AS_OF)
    counts = arxiv.run([source], batches, AS_OF)
    assert (counts["pages"], counts["deleted"]) == (1, 1)
    assert batches.removed == ["https://arxiv.org/abs/hep-th/9901001"]
    batches.close(dict(counts))
    row = rows_of(tmp_path / "run")[0][0]
    assert row["url"] == "https://arxiv.org/abs/1303.7276"
    assert row["text"].splitlines() == [
        "The effect of limb kinematics",
        "Chen Li",
        "arXiv:1303.7276 · physics.bio-ph cs.RO · Submitted 2013-03-29, updated 2026-10-26",
        "## Abstract",
        "Legged robots on sand.",
    ]
    assert (row["published"], row["modified"]) == ("2013-03-29", "2026-10-26")


def oai_record(arxiv_id: str, licence: str) -> str:
    return f"""<record><header><identifier>oai:arXiv.org:{arxiv_id}</identifier></header><metadata>
<arXiv xmlns="http://arxiv.org/OAI/arXiv/"><id>{arxiv_id}</id><created>2024-01-02</created>
<authors><author><keyname>Tian</keyname><forenames>Zehua</forenames></author></authors>
<title>Probing cosmic strings</title><categories>gr-qc</categories>{licence}<abstract>Short.</abstract></arXiv></metadata></record>"""


COMMON_PILE_TEXT = """# Probing cosmic strings

Zehua Tian, Email: tianzh@ustc.edu.cn CAS Key Laboratory of
Microscale Magnetic Resonance, Hefei 230026, China

###### Abstract

Quantum metrology studies the ultimate precision limit
of physical quantities.

CAS Key Laboratory, Hefei 230026, China (tianzh@ustc.edu.cn)

##  1 Introduction

Cosmic strings are line-like
defects. """ + "They bend light. " * 120 + """

######  Lemma 1

The deficit angle is small. Fig. 3\\. shows $\\Delta_{12}$.

  * first point
  * second point

## References

  * (1) M. S. Turner, Science 262, 861 (1993).
"""


def test_arxiv_full_text_is_taken_for_openly_licensed_papers_onto_their_html_view(tmp_path):
    by = "<license>http://creativecommons.org/licenses/by/4.0/</license>"
    oai = tmp_path / "oai"
    oai.mkdir()
    (oai / "oai-000001.xml").write_text(
        '<OAI-PMH xmlns="http://www.openarchives.org/OAI/2.0/"><ListRecords>'
        + oai_record("2401.00001", by)
        + oai_record("2208.05351", "<license>http://creativecommons.org/publicdomain/zero/1.0/</license>")
        + oai_record("2402.00002", "<license>http://arxiv.org/licenses/nonexclusive-distrib/1.0/</license>")
        + oai_record("2403.00003", "")
        + oai_record("2404.00004", by)
        + "</ListRecords></OAI-PMH>"
    )
    pile = tmp_path / "00000_arxiv-papers.jsonl.gz"
    with gzip.open(pile, "wt", encoding="utf-8") as out:
        for arxiv_id, text in (
            ("2401.00001", COMMON_PILE_TEXT),
            ("2208.05351", "# Isotope effects\n\nV.G. Plekhanov\nCollege, Tallinn\n\nAbstract. " + "Isotopes shift the exciton. " * 80),
            ("2402.00002", COMMON_PILE_TEXT),
            ("2403.00003", COMMON_PILE_TEXT),
            ("2404.00004", "# Probing cosmic strings\n\nToo short."),
            ("2405.00005", COMMON_PILE_TEXT),
        ):
            out.write(json.dumps({"id": arxiv_id, "text": text}) + "\n")
    papers = arxiv.open_papers(sorted(oai.glob("*.xml")))
    assert sorted(papers) == ["2208.05351", "2401.00001", "2404.00004"]
    batches = Batches(tmp_path / "run", "arxiv-text", "20261110", AS_OF)
    counts = arxiv.run_text([pile], papers, batches, AS_OF)
    assert (counts["pages"], counts["not_open"], counts["short"]) == (2, 3, 1)
    batches.close(dict(counts))
    rows = {r["url"]: r for r in rows_of(tmp_path / "run")[0]}
    assert sorted(rows) == [
        "https://ar5iv.labs.arxiv.org/html/2208.05351",
        "https://arxiv.org/html/2401.00001",
    ]
    row = rows["https://arxiv.org/html/2401.00001"]
    assert row["text"].splitlines() == [
        "Probing cosmic strings",
        "Zehua Tian",
        "arXiv:2401.00001 · gr-qc · Submitted 2024-01-02",
        "## Abstract",
        "Quantum metrology studies the ultimate precision limit of physical quantities.",
        "## 1 Introduction",
        "Cosmic strings are line-like defects. " + ("They bend light. " * 120).strip(),
        "Lemma 1",
        "The deficit angle is small. Fig. 3. shows $\\Delta_{12}$.",
        "* first point",
        "* second point",
    ]
    assert (row["page_type"], row["content_type"], row["published"]) == ("paper", "text/html", "2024-01-02")
    assert rows["https://ar5iv.labs.arxiv.org/html/2208.05351"]["text"].splitlines()[3].startswith(
        "Abstract. Isotopes shift the exciton."
    ), "without an abstract heading, the text starts at the paragraph that opens with Abstract"


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
    framed, disambiguation = wikipedia.page_html(
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
    assert wikipedia.og_image(small, host).endswith(
        "Sciuridae.jpg?utm_source=en.wikipedia.org&utm_campaign=index&utm_content=thumbnail_unscaled"
    )
    assert wikipedia.og_image(tall, host).startswith(
        "https://thumb.wikimedia.org/wikipedia/commons/thumb/4/43/Sir_Robert.jpg/960px-Sir_Robert.jpg?"
    )
    assert (
        "/thumb/1/18/ISO_C%2B%2B_Logo.svg/1280px-ISO_C%2B%2B_Logo.svg.png?"
        in wikipedia.og_image(vector, host)
    )


def test_a_repeated_article_keeps_its_highest_revision_and_disambiguation_pages_stay_out(
    tmp_path,
):
    snapshot = tmp_path / "enwiki_namespace_0.ndjson"
    disambiguation = article(9, "<p>Squirrel may refer to many things.</p>") | {
        "identifier": 7,
        "url": "https://en.wikipedia.org/wiki/Squirrel_(disambiguation)",
    }
    disambiguation["article_body"]["html"] = disambiguation["article_body"][
        "html"
    ].replace("<title>", '<meta property="mw:PageProp/disambiguation"/><title>')
    lines = [
        article(5, SQUIRREL),
        article(3, SQUIRREL),
        article(8, SQUIRREL.replace("bushy", "long")),
        disambiguation,
    ]
    snapshot.write_text("".join(json.dumps(a) + "\n" for a in lines))
    for workers in (1, 2):
        batches = Batches(tmp_path / f"run-{workers}", "wikipedia", "20261101", AS_OF)
        counts = wikipedia.run([snapshot], batches, AS_OF, set(), workers=workers)
        batches.close(dict(counts))
        rows, _ = rows_of(tmp_path / f"run-{workers}")
        assert (counts["pages"], counts["older_copy"], counts["disambiguation"]) == (
            2,
            1,
            1,
        )
        assert [(r["fetched_at"] - AS_OF).total_seconds() for r in rows] == [0, 1], (
            "the later, higher revision wins on fetch time"
        )
        assert "The tail is long." in rows[1]["text"] and rows[1]["page_type"] == "article"


def test_batch_files_hold_the_upload_schema(tmp_path):
    batches = Batches(tmp_path / "run", "arxiv", "x", AS_OF, rows_per_file=1)
    source = tmp_path / "oai.xml"
    source.write_text(ARXIV)
    arxiv.run([source, source], batches, AS_OF)
    batches.close({})
    manifest = json.loads((tmp_path / "run" / MANIFEST).read_text())
    assert manifest["files"] == ["arxiv-000001.parquet", "arxiv-000002.parquet"]
    assert (
        pq.read_schema(tmp_path / "run" / "arxiv-000001.parquet").remove_metadata()
        == PAGE_SCHEMA.remove_metadata()
    )
