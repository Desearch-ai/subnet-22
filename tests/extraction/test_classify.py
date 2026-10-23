import pytest

from desearch.extraction import extract

PROSE = "The council approved the new transit plan after a long public debate about fares and routes. "


def page(body: str, head: str = "<title>Example</title>") -> str:
    return f"<html><head>{head}</head><body>{body}</body></html>"


def ld(type_: str, extra: str = "") -> str:
    return f'<script type="application/ld+json">{{"@type": "{type_}"{extra}}}</script>'


FRONT = page(
    "<main>"
    + "".join(
        f"<div><h3><a href='/news/s{i}'>Headline {i}</a></h3><a href='/news/s{i}'><p>{PROSE * 2}</p></a></div>"
        for i in range(10)
    )
    + "</main>"
)
ARTICLE = page(
    f"<article><h1>Plan passes</h1><p>{PROSE * 3}</p><p>{PROSE * 3}</p><p>{PROSE * 3}</p></article>",
    "<title>Plan passes</title>" + ld("NewsArticle", ', "datePublished": "2026-10-20"'),
)
TALK = "".join(f"<p>{PROSE * 4}</p>" for _ in range(4))
LINKS = page(
    "".join(f"<a href='/toy/{i}'>Wooden toy number {i}</a> " for i in range(40))
)
POSTS = page(
    "".join(f"<article><h2>Post {i}</h2><p>{PROSE * 3}</p></article>" for i in range(3))
)


@pytest.mark.parametrize(
    ("url", "html", "expected"),
    [
        ("https://news.example/", page("<p>Welcome</p>"), "home"),
        ("https://news.example/?s=transit", page("<p>Results</p>"), "listing"),
        ("https://en.wikipedia.org/wiki/Main_Page", page("<p>Welcome</p>"), "home"),
        (
            "https://news.example/privacy-policy",
            page("<p>We collect data.</p>"),
            "other",
        ),
        ("https://news.example/imprint", page("<p>Company details.</p>"), "other"),
        ("https://news.example/news/world", FRONT, "listing"),
        ("https://news.example/news/world/2026/10/20/plan-passes", ARTICLE, "article"),
        (
            "https://news.example/category/politics/page/2",
            page("<p>Stories</p>"),
            "listing",
        ),
        (
            "https://forum.example/t/brake-noise/123",
            page(f"<p>{PROSE}</p>", ld("DiscussionForumPosting")),
            "forum",
        ),
        (
            "https://shop.example/pine-shelf",
            page("<p>$49.99</p><button>Add to cart</button>"),
            "product",
        ),
        ("https://shop.example/p/1", page("<p>Shelf</p>", ld("Product")), "product"),
        (
            "https://docs.example/docs/v2/install",
            page(f"<p>{PROSE * 2}</p><p>On this page</p>"),
            "docs",
        ),
        (
            "https://acme.example/about-us",
            page(f"<p>{PROSE}</p>", ld("Article", ', "datePublished": "2026-10-20"')),
            "company",
        ),
        (
            "https://acme.example/pricing",
            page("<p>Plans for every team.</p>"),
            "landing",
        ),
        (
            "https://museum.example/events/quilt-talk",
            page(TALK),
            "other",
        ),
        (
            "https://museum.example/whats-on/kentucky-quilts-and-quiltmakers",
            page(TALK, ld("Event")),
            "other",
        ),
        (
            "https://news.example/events/2026/10/20/plan-passes",
            ARTICLE,
            "article",
        ),
        ("https://radio.example/blog/2021/08/", POSTS, "listing"),
        ("https://think.example/category/legal/", POSTS, "listing"),
        ("https://mag.example/latest", page("<p>Stories</p>"), "listing"),
        ("https://blog.example/sitemap.html", page("<p>Pages</p>"), "listing"),
        ("https://shop.example/products/toys", LINKS, "listing"),
        ("https://news.example/news/latest?tag=transit", POSTS, "listing"),
    ],
)
def test_page_types(url, html, expected):
    assert extract(html, url).page_type == expected


def test_long_paragraphs_inside_links_do_not_make_a_front_an_article():
    assert extract(FRONT, "https://news.example/news/world").text_mode == "full"
