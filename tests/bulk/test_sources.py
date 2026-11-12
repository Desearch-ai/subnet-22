import json
from datetime import datetime, timezone

import pyarrow.parquet as pq
from lxml import etree

from bulk import wikipedia
from bulk.rows import MANIFEST, Batches, upload_row
from desearch.extraction import Page
from desearch.extraction.schema import PAGE_SCHEMA

AS_OF = datetime(2026, 11, 1, tzinfo=timezone.utc)


def rows_of(out):
    manifest = json.loads((out / MANIFEST).read_text())
    return [
        r for name in manifest["files"] for r in pq.read_table(out / name).to_pylist()
    ], manifest


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
    batches = Batches(tmp_path / "run", "wikipedia", "x", AS_OF, rows_per_file=1)
    for n in (1, 2):
        batches.add(upload_row(f"https://en.wikipedia.org/wiki/{n}", Page(title=str(n), text="t"), AS_OF, "text/html"))
    batches.close({})
    manifest = json.loads((tmp_path / "run" / MANIFEST).read_text())
    assert manifest["files"] == ["wikipedia-000001.parquet", "wikipedia-000002.parquet"]
    assert (
        pq.read_schema(tmp_path / "run" / "wikipedia-000001.parquet").remove_metadata()
        == PAGE_SCHEMA.remove_metadata()
    )
