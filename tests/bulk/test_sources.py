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

JATS = """<article xml:lang="en"><front><article-meta><title-group><article-title>Risk factors</article-title></title-group>
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
