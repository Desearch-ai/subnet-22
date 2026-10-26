import re
from datetime import datetime, timezone

import pyarrow as pa
import pytest

from desearch.extraction.schema import PAGE_SCHEMA
from desearch.fetch import Fetched
from neurons.miners.rows import build_row, write_parquet
from neurons.validators.scoring import (
    FetchedPage,
    derived_from_html,
    judge_sample,
    load_upload,
    read_pages,
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


def test_an_upload_written_by_the_miner_is_re_extracted_equal_and_passes():
    html = story()
    fetched = Fetched(
        url=STORY_URL,
        final_url=STORY_URL,
        fetched_at=datetime.now(timezone.utc),
        status=200,
        content_type="text/html",
        body=html.encode(),
    )
    row = build_row(fetched, 5_000_000)
    upload = write_parquet([row], "t", "hk")
    rows, _ = load_upload(upload, [STORY_URL], SEED)

    assert rows[0]["extraction"] == "v2"
    assert rows[0]["text_mode"] == "main" and rows[0]["paywalled"] is True
    assert derived_from_html(rows[0])
    result = run(rows, [STORY_URL], {STORY_URL: FetchedPage(200, html)})
    assert (result["verdict"], result["reason"]) == ("pass", "ok")


def test_a_page_redirected_to_another_site_re_extracts_equal():
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

    row = build_row(fetched, 5_000_000)

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
def test_a_forged_page_field_is_caught_by_re_extraction(field, value):
    row = page_row(STORY_URL, story())
    assert derived_from_html(row) and row[field] != value

    assert not derived_from_html(dict(row, **{field: value}))


def test_a_file_missing_a_column_or_naming_another_extraction_is_unreadable():
    rows, _, _ = synthetic(3)
    no_ld = pa.schema([field for field in PAGE_SCHEMA if field.name != "ld"])
    relabelled = [rows[0], dict(rows[1], extraction="other"), rows[2]]

    assert read_pages(to_parquet(rows, no_ld)) is None
    assert read_pages(to_parquet(relabelled)) is None
    assert read_pages(to_parquet(rows)) is not None


def test_an_empty_published_proves_nothing_only_where_it_cannot_be_forged():
    dated = page_row(STORY_URL, story())
    undated = page_row(STORY_URL, story(published=""))
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
    row = page_row(STORY_URL, story())
    changed = story(
        modified="2026-10-01T08:00:00+00:00",
        image="https://cdn.example/other.jpg",
        robots="index",
    )

    sample = judge_sample(row, FetchedPage(200, changed))

    assert (sample["outcome"], sample["fields_differ"]) == ("matched", [])


@pytest.mark.parametrize(
    "html, url, outcome",
    [(board, BOARD_URL, "matched"), (story, STORY_URL, "mismatched")],
)
def test_full_text_pages_get_the_churn_allowance(html, url, outcome):
    row = page_row(url, html())
    swapped = html(seeds=[*range(8), 20, 21])

    sample = judge_sample(row, FetchedPage(200, swapped))

    assert sample["outcome"] == outcome, sample["why"]
