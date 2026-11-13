import json
import os

from lxml import etree


def load_worker():
    # The worker points stdout at stderr on import; the test process keeps its own.
    saved = os.dup(1)
    try:
        from bulk import page_worker
    finally:
        os.dup2(saved, 1)
        os.close(saved)
    return page_worker


page_worker = load_worker()


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
    framed, disambiguation = page_worker.page_html(
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
    assert page_worker.og_image(small, host).endswith(
        "Sciuridae.jpg?utm_source=en.wikipedia.org&utm_campaign=index&utm_content=thumbnail_unscaled"
    )
    assert page_worker.og_image(tall, host).startswith(
        "https://thumb.wikimedia.org/wikipedia/commons/thumb/4/43/Sir_Robert.jpg/960px-Sir_Robert.jpg?"
    )
    assert (
        "/thumb/1/18/ISO_C%2B%2B_Logo.svg/1280px-ISO_C%2B%2B_Logo.svg.png?"
        in page_worker.og_image(vector, host)
    )


def test_disambiguation_pages_are_marked():
    disambiguation = article(9, "<p>Squirrel may refer to many things.</p>")
    disambiguation["article_body"]["html"] = disambiguation["article_body"][
        "html"
    ].replace("<title>", '<meta property="mw:PageProp/disambiguation"/><title>')
    assert page_worker.page_html(disambiguation, set())[1]
    assert not page_worker.page_html(article(9, SQUIRREL), set())[1]
