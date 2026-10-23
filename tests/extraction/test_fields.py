import json

from desearch.extraction import Page, extract
from desearch.extraction.schema import PAGE_SCHEMA_V1, PAGE_SCHEMA_V2
from desearch.extraction.structured.dates import read_date

HEAD = """<title>Pine shelf</title>
<meta property="og:site_name" content="Woodshop">
<meta property="og:image" content="/img/shelf.jpg">
<meta property="article:section" content="Furniture">
<meta property="article:tag" content="pine"><meta property="article:tag" content="shelving">
<meta name="news_keywords" content="diy, pine">
<meta property="article:modified_time" content="2026-10-21T08:00:00Z">
<meta name="robots" content="max-snippet:-1, NOINDEX">
<script type="application/ld+json">{"@graph": [
 {"@type": "Article", "datePublished": "2026-10-20T23:30:00-05:00", "isAccessibleForFree": "False",
  "keywords": ["shelving", "joinery"], "wordCount": 1200, "author": {"name": "Sam Wood"}},
 {"@type": "BreadcrumbList", "itemListElement": [
   {"@type": "ListItem", "position": 2, "name": "Furniture"}, {"@type": "ListItem", "position": 1, "name": "Home"}]},
 {"@type": "Organization", "name": "Woodshop Ltd", "url": "https://woodshop.example/",
  "sameAs": ["https://x.com/woodshop"], "address": {"addressLocality": "Austin", "addressCountry": "US"}},
 {"@type": "FAQPage", "mainEntity": [{"@type": "Question", "name": "Is pine strong?",
   "acceptedAnswer": {"text": "<p>Yes, for shelves.</p>"}}]},
 {"@type": "Product", "name": "Pine shelf", "sku": "PS-1", "offers": {"price": 49.5, "priceCurrency": "USD",
   "availability": "https://schema.org/InStock"}}
]}</script>"""


def test_fields():
    got = extract(
        f"<html lang='en-GB'><head>{HEAD}</head><body><p>Shelves.</p></body></html>",
        "https://woodshop.example/blog/pine-shelf",
    )
    assert (got.published, got.modified) == (
        "2026-10-21T04:30:00Z",
        "2026-10-21T08:00:00Z",
    )
    assert (got.site_name, got.image, got.section, got.author) == (
        "Woodshop",
        "https://woodshop.example/img/shelf.jpg",
        "Furniture",
        "Sam Wood",
    )
    assert got.tags == ["pine", "shelving", "diy", "joinery"]
    assert got.breadcrumbs == ["Home", "Furniture"]
    assert got.paywalled is True and got.robots == ["max-snippet:-1", "noindex"]
    assert json.loads(got.entity) == {
        "type": "Organization",
        "name": "Woodshop Ltd",
        "url": "https://woodshop.example/",
        "same_as": ["https://x.com/woodshop"],
        "address": {"locality": "Austin", "country": "US"},
    }
    ld = json.loads(got.ld)
    assert ld["faq"] == [{"q": "Is pine strong?", "a": "Yes, for shelves."}]
    assert ld["product"] == {
        "name": "Pine shelf",
        "sku": "PS-1",
        "price": "49.5",
        "currency": "USD",
        "availability": "InStock",
    }
    assert ld["word_count"] == "1200"
    assert got.entity == json.dumps(
        json.loads(got.entity),
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    )


def test_dates():
    assert read_date("2026-10-20T23:30:00-05:00") == "2026-10-21T04:30:00Z"
    assert read_date("2026-10-20T23:30:00") == "2026-10-20"
    assert read_date("2026-10-20") == "2026-10-20"
    assert (
        read_date("2026-02-30") == ""
        and read_date("not a date") == ""
        and read_date(None) == ""
    )
    html = "<html><head><meta property='article:published_time' content='2026-10-20'><meta property='article:modified_time' content='2026-10-01'></head><body></body></html>"
    assert extract(html, "https://a.example/x").modified == ""


def test_v2_schema_extends_v1_and_carries_every_page_field():
    assert list(PAGE_SCHEMA_V2)[: len(PAGE_SCHEMA_V1)] == list(PAGE_SCHEMA_V1)
    page_fields = set(Page.__dataclass_fields__)
    assert page_fields <= set(PAGE_SCHEMA_V2.names)
    assert set(PAGE_SCHEMA_V2.names) - set(PAGE_SCHEMA_V1.names) == {
        "extraction",
        "text_mode",
        "modified",
        "site_name",
        "image",
        "section",
        "tags",
        "breadcrumbs",
        "entity",
        "ld",
        "paywalled",
        "robots",
    }


def test_odd_json_ld_and_attributes_never_empty_the_page():
    head = (
        '<title>T</title><script type="application/ld+json">{"@graph": ['
        '{"@type": "NewsArticle", "author": {"name": "Ann \\ud800Lee"}, "keywords": ["a\\udc00b"]},'
        '{"@type": "BreadcrumbList", "itemListElement": [{"position": Infinity, "name": "A"},'
        ' {"position": NaN, "name": "B"}, {"position": "²", "name": "C"}]}]}</script>'
    )
    body = (
        "<table><tr><td colspan='²'>a</td><td>b</td></tr></table><p>Some text here.</p>"
    )
    got = extract(
        f"<html><head>{head}</head><body>{body}</body></html>",
        "https://a.example/x?page=²",
    )
    assert (got.author, got.tags, got.breadcrumbs) == (
        "Ann Lee",
        ["ab"],
        ["A", "B", "C"],
    )
    assert "| a | b |" in got.text and "Some text here." in got.text
