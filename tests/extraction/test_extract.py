import pytest

from desearch.extraction import FULL, extract, page_text
from desearch.extraction.classify import classify
from desearch.extraction.structured import read_structure
from desearch.extraction.structured.html import parse, restore_misnested_svgs
from desearch.extraction.text.page import full_text
from tests.extraction.pages import ARTICLE, PROSE, STORY_URL, page


def test_same_html_gives_the_same_page_and_bytes_work():
    assert extract(ARTICLE, "https://news.example/2026/10/20/x") == extract(
        ARTICLE.encode(), "https://news.example/2026/10/20/x"
    )
    assert (
        extract("", "https://a.example/").text == ""
        and extract("", "https://a.example/").text_mode == FULL
    )


def test_a_page_whose_body_has_a_popup_class_still_extracts():
    html = f"<html><head><title>Plans</title></head><body class='modal-open'><p>{PROSE * 3}</p></body></html>"
    assert "council approved" in extract(html, "https://acme.example/plans").text


def test_page_markup_the_parser_nests_inside_an_svg_sprite_is_put_back():
    html = f"<html><body><svg><symbol id='x'><path d='M0'/></symbol></svg><div><h1>Brake noise</h1><p>{PROSE}</p></div></body></html>"
    root = parse(html)
    # The shape libxml2 leaves when it cannot close a large sprite: the rest of the page inside it.
    root.find(".//symbol").append(root.find(".//div"))
    restore_misnested_svgs(root)
    assert root.find(".//svg") is None and root.find(".//path") is None
    assert "council approved" in full_text(root, "Forum")


@pytest.mark.parametrize(
    "html, url",
    [
        (ARTICLE, STORY_URL),
        (page(f"<h1>Acme</h1><p>{PROSE * 3}</p>"), "https://acme.example/"),
        (
            page(f"<h1>Pine shelf</h1><p>{PROSE}</p><p>$49.99</p>"),
            "https://shop.example/products/pine-shelf",
        ),
    ],
)
def test_every_step_runs_without_the_guard_that_turns_errors_into_empty_pages(
    html, url
):
    root = parse(html)
    meta = read_structure(root, url)
    page_type = classify(url, meta, root)
    text, _, _ = page_text(
        root, url, meta.title, meta.generator, page_type, meta.author
    )
    assert "council approved" in text
    assert extract(html, url).text == text
