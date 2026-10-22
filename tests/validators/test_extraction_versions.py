import re
from datetime import datetime, timezone

import pyarrow as pa
import pytest

from desearch.extraction import EXTRACTION_V1, EXTRACTION_V2
from desearch.extraction.schema import PAGE_SCHEMA, PAGE_SCHEMA_V2
from desearch.fetch import Fetched
from neurons.miners.rows import build_row, write_parquet
from neurons.validators.scoring import (
    V1,
    V2,
    FetchedPage,
    derived_from_html,
    judge_sample,
    load_upload,
    read_pages,
    row_extraction,
)
from tests.synthetic import SEED, page_row, run, synthetic, to_parquet

STORY_URL = "https://daily.example/news/2026/09/22/story"
BOARD_URL = "https://board.example/widgets/blue-widget"


def token(n: int) -> str:
    """A word of letters only, distinct for every n."""
    word = ""
    n += 1
    while n:
        n, rest = divmod(n - 1, 26)
        word = chr(97 + rest) + word
    return word


def paragraphs(seeds) -> str:
    return "".join(
        "<p>" + " ".join(token(seed * 1000 + i) for i in range(40)) + ".</p>"
        for seed in seeds
    )


def story(
    published: str = "2026-09-22T14:05:00+00:00",
    modified: str = "2026-09-23T10:00:00+00:00",
    image: str = "https://cdn.example/a.jpg",
    robots: str = "noarchive",
    seeds=range(10),
) -> str:
    dated = (
        f'<meta property="article:published_time" content="{published}">'
        if published
        else ""
    )
    return (
        "<html><head><title>Story</title>"
        '<meta property="og:type" content="article">'
        f"{dated}"
        f'<meta property="article:modified_time" content="{modified}">'
        f'<meta property="og:image" content="{image}">'
        f'<meta name="robots" content="{robots}">'
        '<meta property="og:site_name" content="Daily">'
        '<script type="application/ld+json">{"@context":"https://schema.org",'
        '"@type":"NewsArticle","headline":"Story","isAccessibleForFree":false,'
        '"author":{"@type":"Person","name":"Ann Lee"}}</script>'
        f"</head><body><article><h1>Story</h1>{paragraphs(seeds)}</article>"
        "</body></html>"
    )


def board(seeds=range(10)) -> str:
    return (
        "<html><head><title>Board</title></head><body>"
        f"<div>{paragraphs(seeds)}</div></body></html>"
    )


def test_a_v2_upload_written_by_the_miner_is_re_extracted_equal_and_passes():
    html = story()
    fetched = Fetched(
        url=STORY_URL,
        final_url=STORY_URL,
        fetched_at=datetime.now(timezone.utc),
        status=200,
        content_type="text/html",
        body=html.encode(),
    )
    row = build_row(fetched, 5_000_000, EXTRACTION_V2)
    upload = write_parquet([row], "t", "hk", EXTRACTION_V2)
    rows, _ = load_upload(upload, [STORY_URL], SEED)

    assert row_extraction(rows[0]) is V2
    assert rows[0]["text_mode"] == "main" and rows[0]["paywalled"] is True
    assert derived_from_html(rows[0])
    result = run(rows, [STORY_URL], {STORY_URL: FetchedPage(200, html)})
    assert (result["verdict"], result["reason"]) == ("pass", "ok")


@pytest.mark.parametrize("extraction", [EXTRACTION_V1, EXTRACTION_V2])
def test_a_page_redirected_to_another_site_re_extracts_equal(extraction):
    html = story().replace(
        "<title>Story</title>",
        '<title>Story</title><link rel="canonical" href="/story">'
        '<meta property="og:image" content="/a.jpg">',
    )
    fetched = Fetched(
        url=STORY_URL,
        final_url="https://moved.example/story",
        fetched_at=datetime.now(timezone.utc),
        status=200,
        content_type="text/html",
        body=html.encode(),
    )

    row = build_row(fetched, 5_000_000, extraction)

    assert row["canonical"] == "https://daily.example/story"
    assert derived_from_html(row)


@pytest.mark.parametrize(
    "field, value",
    [
        ("text_mode", "full"),
        ("modified", "2020-01-01"),
        ("site_name", "Elsewhere"),
        ("image", "https://spam.example/x.jpg"),
        ("tags", ["injected"]),
        ("entity", '{"name": "Spam"}'),
        ("ld", '{"faq": []}'),
        ("paywalled", None),
        ("robots", []),
        ("published", "2026-09-22"),
    ],
)
def test_a_forged_v2_field_is_caught_by_re_extraction(field, value):
    row = page_row(STORY_URL, story(), extraction=EXTRACTION_V2)
    assert derived_from_html(row) and row[field] != value

    assert not derived_from_html(dict(row, **{field: value}))


def test_a_v2_file_reads_as_v2_and_v2_rows_in_a_v1_file_are_checked_as_v1():
    rows, assigned, fetched = synthetic(10, extraction=EXTRACTION_V2)

    read, _ = load_upload(to_parquet(rows, PAGE_SCHEMA_V2), assigned, SEED)
    assert [row_extraction(row) for row in read] == [V2] * 10
    assert run(read, assigned, fetched)["verdict"] == "pass"

    v2_row = page_row(STORY_URL, story(), extraction=EXTRACTION_V2)
    as_v1, _ = load_upload(to_parquet([v2_row], PAGE_SCHEMA), [STORY_URL], SEED)
    assert row_extraction(as_v1[0]) is V1
    result = run(as_v1, [STORY_URL], {STORY_URL: FetchedPage(200, story())})
    assert (result["verdict"], result["reason"]) == ("fail", "text_not_from_html")


def test_a_file_mixing_the_two_schemas_is_unreadable():
    rows, _, _ = synthetic(3, extraction=EXTRACTION_V2)
    v1_plus = pa.schema(
        [*PAGE_SCHEMA, PAGE_SCHEMA_V2.field("extraction"), PAGE_SCHEMA_V2.field("ld")]
    )
    relabelled = [rows[0], dict(rows[1], extraction=EXTRACTION_V1), rows[2]]

    assert read_pages(to_parquet(rows, v1_plus)) is None
    assert read_pages(to_parquet(relabelled, PAGE_SCHEMA_V2)) is None
    assert read_pages(to_parquet(rows, PAGE_SCHEMA_V2)) is not None


@pytest.mark.parametrize("extraction", [EXTRACTION_V1, EXTRACTION_V2])
def test_an_empty_published_proves_nothing_only_where_it_cannot_be_forged(extraction):
    dated = page_row(STORY_URL, story(), extraction=extraction)
    undated = page_row(STORY_URL, story(published=""), extraction=extraction)
    bare = re.sub(r"<meta[^>]*>|<script.*?</script>", "", story(published=""))

    miner_undated = judge_sample(undated, FetchedPage(200, story()))
    live_bare = judge_sample(dated, FetchedPage(200, bare))
    live_undated = judge_sample(dated, FetchedPage(200, story(published="")))
    redated = judge_sample(
        dated, FetchedPage(200, story(published="2019-01-01T00:00:00+00:00"))
    )

    for sample in (miner_undated, live_bare):
        assert (
            sample["outcome"] == "unverifiable"
            and "published" in sample["fields_differ"]
        )
        assert "empty on one side" in sample["why"]
    assert live_undated["outcome"] == "mismatched", "a date the live page lacks"
    assert redated["outcome"] == "mismatched"


def test_modified_image_ld_entity_and_robots_are_not_compared_live():
    row = page_row(STORY_URL, story(), extraction=EXTRACTION_V2)
    changed = story(
        modified="2026-10-01T08:00:00+00:00",
        image="https://cdn.example/other.jpg",
        robots="index",
    )

    sample = judge_sample(row, FetchedPage(200, changed))

    assert (sample["outcome"], sample["fields_differ"]) == ("matched", [])


@pytest.mark.parametrize(
    "html, url, extraction, outcome",
    [
        (board, BOARD_URL, EXTRACTION_V2, "matched"),
        (board, BOARD_URL, EXTRACTION_V1, "mismatched"),
        (story, STORY_URL, EXTRACTION_V2, "mismatched"),
    ],
)
def test_full_text_v2_pages_get_the_churn_allowance(html, url, extraction, outcome):
    row = page_row(url, html(), extraction=extraction)
    swapped = html(seeds=[*range(8), 20, 21])

    sample = judge_sample(row, FetchedPage(200, swapped))

    assert sample["outcome"] == outcome, sample["why"]
