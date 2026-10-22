from desearch.extraction import FULL, MAIN, extract
from desearch.extraction.structured.html import parse
from desearch.extraction.text.article import scope
from desearch.extraction.text.dom import MAIN_DROPPED, body_copy
from tests.extraction.pages import (
    ARTICLE,
    ARTICLE_HEAD,
    PROSE,
    STORY_URL,
    page,
    story_page,
)


def test_articles_get_their_main_content_as_markdown():
    got = extract(
        ARTICLE, "https://news.example/2026/10/20/council-approves-transit-plan"
    )
    assert (got.page_type, got.text_mode) == ("article", MAIN)
    lines = got.text.splitlines()
    assert (
        lines[:2]
        == ["Council approves transit plan", "# Council approves transit plan"]
        or lines[0] == "# Council approves transit plan"
    )
    assert (
        "Other story one" not in got.text
        and "Privacy" not in got.text
        and "Sport" not in got.text
    )
    assert got.headings == ["Council approves transit plan"]


def test_a_short_scope_beside_a_long_page_falls_back_to_full_content():
    html = page(
        f"<main><p>{PROSE * 2}</p></main>"
        + "".join(
            f"<section><h2>Part {i}</h2><div>Section {i}. {PROSE * 2}</div></section>"
            for i in range(6)
        ),
        ARTICLE_HEAD,
    )
    got = extract(html, "https://news.example/2026/10/20/council-approves-transit-plan")
    assert got.text_mode == FULL and "## Part 5" in got.text


def test_mediawiki_articles():
    html = (
        "<html><head><title>Ada Lovelace - Wikipedia</title><meta name='generator' content='MediaWiki 1.47.0'></head><body>"
        "<h1 id='firstHeading'>Ada Lovelace</h1>"
        "<div class='shortdescription'>English mathematician (1815–1852)</div>"
        "<div id='mw-content-text'><div class='mw-parser-output'>"
        "<table class='infobox'><tr><th>Born</th><td>10 December 1815<br>London</td></tr>"
        "<tr><th>Known for</th><td><ul><li>Analytical Engine</li><li>First program</li></ul></td></tr></table>"
        f"<p>Ada Lovelace was a mathematician.<sup class='reference'><a href='#cite_note-1'>[1]</a></sup> {PROSE}</p>"
        "<div class='mw-heading mw-heading2'><h2>Work</h2><span class='mw-editsection'>edit</span></div>"
        f"<p>{PROSE}</p>"
        "<div class='mw-heading mw-heading2'><h2>References</h2></div><ol><li>A cited book</li></ol>"
        "</div></div><div id='mw-normal-catlinks'><ul><li><a href='/c'>English mathematicians</a></li></ul></div>"
        "</body></html>"
    )
    got = extract(html, "https://en.wikipedia.org/wiki/Ada_Lovelace")
    assert got.text_mode == MAIN and got.tags == ["English mathematicians"]
    assert (
        "English mathematician (1815–1852)" in got.text
        and "Known for: Analytical Engine, First program" in got.text
    )
    assert "Born: 10 December 1815 London" in got.text and "## Work" in got.text
    assert (
        "[1]" not in got.text
        and "edit" not in got.text
        and "A cited book" not in got.text
    )


def test_link_boxes_and_tags_after_the_body_are_dropped():
    body = (
        f"<h1>Council approves transit plan</h1><p>{PROSE * 4}</p><p>The plan starts next year.</p>"
        "<div><span>More on this story</span><a href='/news/a'><h2>Fares to rise</h2></a>"
        "<a href='/news/b'><h2>Routes redrawn</h2></a></div>"
        "<div><span>Related internet links</span><a href='https://transit.example/'><h2>Transit Agency</h2></a></div>"
        "<ul><li><a href='/topics/city'>City</a></li><li><a href='/topics/transport'>Transport</a></li></ul>"
    )
    text = extract(story_page(body), STORY_URL).text
    assert text.endswith("The plan starts next year.")


def test_header_beside_the_article_is_kept():
    header = (
        "<header><h1>Council approves transit plan</h1><p class='dek'>Fares hold steady for a year.</p>"
        "<div class='byline'>By <a href='/authors/jane'>Jane Doe</a> <time datetime='2026-10-20'>Oct 20</time></div></header>"
    )
    html = page(
        f"<main>{header}<article><p>{PROSE * 4}</p></article></main>", ARTICLE_HEAD
    )
    text = extract(html, STORY_URL).text
    assert (
        "# Council approves transit plan" in text
        and "Fares hold steady for a year." in text
    )
    assert "By Jane Doe Oct 20" in text


def test_a_byline_with_links_survives_the_link_rules():
    byline = (
        "<div><a href='/authors/jane'>Jane Doe</a><a href='https://x.example/jane'>X</a>"
        "<a href='/section/city'>City</a><time datetime='2026-10-20'>Oct 20</time></div>"
    )
    text = extract(
        story_page(f"<h1>Council approves transit plan</h1>{byline}<p>{PROSE * 4}</p>"),
        STORY_URL,
    ).text
    assert "Jane Doe" in text and "Oct 20" in text


def test_widgets_are_found_by_the_words_of_their_names():
    body = (
        f"<h1>Council approves transit plan</h1><p>{PROSE * 4}</p>"
        "<div class='RelatedArticles-module__box'><p>Another story you might like today.</p></div>"
        "<div class='commentCount'>12 comments</div><div class='ad-slot'>Buy now</div>"
        f"<div class='show-next-ad'><p>{PROSE}</p></div><p>Advertisement</p>"
    )
    text = extract(story_page(body), STORY_URL).text
    assert (
        "Another story" not in text
        and "12 comments" not in text
        and "Buy now" not in text
    )
    assert "Advertisement" not in text and text.count(PROSE.strip()) == 5


def test_empty_headings_go_but_kicker_headings_stay():
    body = (
        f"<h1>Council approves transit plan</h1><h3>Best for commuters</h3><h2>Route 9</h2><p>{PROSE * 4}</p>"
        "<h2>Most read</h2><ul><li><a href='/a'>One</a></li><li><a href='/b'>Two</a></li><li><a href='/c'>Three</a></li></ul>"
    )
    text = extract(story_page(body), STORY_URL).text
    assert "### Best for commuters" in text and "Most read" not in text


def test_pages_without_article_or_main_use_the_element_holding_the_prose():
    halves = (
        f"<div class='post'><h1>Council approves transit plan</h1><div><p>{PROSE * 4}</p><p>Fares stay flat. {PROSE * 4}</p></div>"
        "<form class='newsletter'><p>Sign up for our newsletter</p></form>"
        f"<div><p>Second half. {PROSE * 4}</p></div></div>"
    )
    html = page(
        f"<div class='site'><div class='menu'><a href='/'>Home</a></div>{halves}</div>",
        ARTICLE_HEAD,
    )
    got = extract(html, STORY_URL)
    assert (
        got.text_mode == MAIN
        and "Second half." in got.text
        and "newsletter" not in got.text
    )


def test_docs_pages_keep_their_link_lists():
    body = (
        "<h1>Install</h1><p>" + PROSE * 4 + "</p><h2>See also</h2>"
        "<ul><li><a href='/docs/pagination'>Pagination</a></li><li><a href='/docs/search'>Search</a></li></ul>"
        "<section id='pagination'><h2>Pagination</h2><p>Pass page=2 to get the next page of results.</p></section>"
    )
    got = extract(page(f"<main>{body}</main>"), "https://docs.example/docs/v2/install")
    assert got.page_type == "docs" and "- Pagination" in got.text
    assert "Pass page=2 to get the next page of results." in got.text


def test_lists_inside_an_article_stay_but_link_boxes_at_its_ends_go():
    cast = "".join(
        f"<div><a href='/people/{i}'>Actor {i}</a></div><div>Role {i}</div>"
        for i in range(3)
    )
    downloads = "<table><tr><td><p><a href='/reports/7/file'>Annual report (pdf, 2 MB)</a></p></td></tr></table>"
    body = (
        "<p><a href='/section/city'>City</a></p><h1>Council approves transit plan</h1>"
        f"<p>{PROSE * 3}</p>{cast}<p>{PROSE * 3}</p>{downloads}"
        "<div><a href='/news/older'>An older story about the council</a></div>"
    )
    text = extract(story_page(body), STORY_URL).text
    assert "Actor 1" in text and "Annual report (pdf, 2 MB)" in text
    assert "older story" not in text and "\nCity\n" not in text


def test_bylines_by_name_and_trailing_link_runs():
    byline = "<p>By <a href='/person/jane'>Jane Doe</a> and <a href='/person/sam'>Sam Roe</a></p>"
    tags = "<h3>Topics</h3><ul><li><a href='/t/a'>Alpha</a></li><li><a href='/t/b'>Beta</a></li></ul>"
    text = extract(
        story_page(f"<h1>Plan</h1>{byline}<p>{PROSE * 4}</p>{tags}"), STORY_URL
    ).text
    assert "By Jane Doe and Sam Roe" in text and text.endswith(PROSE.strip())


def test_calls_to_action_at_the_ends_of_an_article_go():
    follow = (
        "<div><span><strong>Follow topics and authors</strong> from this story to see more like this.</span>"
        "<ul><li><span role='button'>Jane Doe</span></li></ul></div>"
    )
    body = (
        "<h1>Council approves transit plan</h1><p>Add City News on Google</p>"
        f"<p>{PROSE * 3}</p><p>Follow the money, critics say, and the plan looks different. {PROSE}</p>"
        f"<p>{PROSE * 3}</p>{follow}"
    )
    text = extract(story_page(body), STORY_URL).text
    assert (
        "Follow topics" not in text
        and "Jane Doe" not in text
        and "on Google" not in text
    )
    assert "Follow the money, critics say" in text


def test_a_short_article_on_a_busy_page_keeps_its_main_text():
    story = (
        f"<article><h1>Plan passes</h1><p>{PROSE * 4}</p><p>{PROSE * 3}</p></article>"
    )
    teasers = "".join(
        f"<div class='promo'><h3>Other story {i}</h3>District {i} will vote on its school, park and road budget next spring.</div>"
        for i in range(18)
    )
    html = f"<html><head><title>Plan passes</title></head><body>{story}{teasers}</body></html>"
    page = extract(html, "https://news.example/news/2026/10/20/plan-passes")
    assert page.text_mode == MAIN
    assert "Other story" not in page.text


def test_an_article_written_in_divs_is_scoped_to_its_article_element():
    body = "".join(f"<div>Part {i}. {PROSE * 2}</div>" for i in range(6))
    related = "".join(
        f"<div><a href='/s/{i}'>Related story {i}</a></div>" for i in range(6)
    )
    html = f"<html><body><article><h1>Plan passes</h1>{body}</article>{related}</body></html>"
    found = scope(body_copy(parse(html), MAIN_DROPPED), article=True)
    assert found is not None and found.tag == "article"


def test_articles_keep_faq_questions_but_drop_buttons():
    faq = "".join(
        f"<button aria-expanded='false'>Question {i}: does the plan cover buses?</button><p>Answer {i}. {PROSE}</p>"
        for i in range(2)
    )
    html = (
        '<html><head><title>Plan passes</title><script type="application/ld+json">'
        '{"@type": "NewsArticle", "datePublished": "2026-10-20"}</script></head><body>'
        f"<article><h1>Plan passes</h1><p>{PROSE * 4}</p><button>Share</button><p>{PROSE * 3}</p>{faq}</article>"
        "</body></html>"
    )
    page = extract(html, "https://news.example/news/2026/10/20/plan-passes")
    assert page.text_mode == MAIN
    assert "Question 1: does the plan cover buses?" in page.text
    assert "Share" not in page.text


def test_articles_drop_the_author_bio_and_audio_player_lines():
    head = (
        "<title>Plan passes</title><meta name='author' content='Jon Brodkin'>"
        '<script type="application/ld+json">{"@type": "NewsArticle", "datePublished": "2026-10-20"}</script>'
    )
    body = (
        "<article><h1>Plan passes</h1><p>Getting your Trinity Audio player ready...</p>"
        f"<p>{PROSE * 4}</p><p>The vote ends a decade of debate over the routes.</p>"
        "<p>Jon is a Senior IT Reporter for Ars Technica. He covers the telecom industry.</p>"
        "<p>Reach him at jon@example.com</p></article>"
    )
    page = extract(
        f"<html><head>{head}</head><body>{body}</body></html>",
        "https://news.example/news/2026/10/20/plan-passes",
    )
    assert page.text_mode == MAIN
    assert "The vote ends a decade of debate" in page.text
    for gone in (
        "Trinity Audio",
        "Senior IT Reporter",
        "telecom industry",
        "jon@example.com",
    ):
        assert gone not in page.text


def test_articles_keep_a_wrapped_byline_and_drop_appeal_boxes():
    head = (
        "<title>Renters gain ground in San Antonio | Report</title><meta name='author' content='Jasper Sundeen'>"
        '<script type="application/ld+json">{"@type": "NewsArticle", "datePublished": "2026-09-13"}</script>'
    )
    html = (
        f"<html><head>{head}</head><body><main>"
        "<header><h1>San Antonio has a renter-friendly market now</h1>"
        "<div class='entry-subhead'><span class='byline'>by <a href='/author/jasper'>Jasper Sundeen</a></span>"
        " <time>September 13, 2026</time> <a href='/trust'>Why you can trust us</a></div></header>"
        f"<div class='main-content'><article><p>{PROSE * 4}</p>"
        "<div><p>Keep our news free from ads and paywalls by making a donation to support our work!</p>"
        "<p>We are run by a small editorial team.</p></div>"
        f"<p>{PROSE * 3}</p><p>Subscribe to our newsletter for weekly updates.</p></article></div>"
        "</main></body></html>"
    )
    page = extract(html, "https://news.example/2026/09/13/renters")
    assert page.text_mode == MAIN
    assert "by Jasper Sundeen" in page.text and "September 13, 2026" in page.text
    for gone in (
        "Keep our news free",
        "small editorial team",
        "Subscribe to our newsletter",
    ):
        assert gone not in page.text


def test_articles_without_article_or_main_are_scoped_to_their_named_content_element():
    post = "".join(f"<p>Part {i}. {PROSE * 2}</p>" for i in range(6))
    html = (
        '<html><head><title>Plan passes</title><script type="application/ld+json">'
        '{"@type": "NewsArticle", "datePublished": "2026-10-20"}</script></head><body>'
        "<div class='layout'><div class='masthead-promo'><p>Shop our store for gifts and more ideas today.</p></div>"
        "<div class='post-134 post type-post status-publish tag-advertising tag-popular entry-content no-share'>"
        f"<h1>Plan passes</h1>{post}</div>"
        f"<div class='sidebar-box'><p>{PROSE}</p></div></div></body></html>"
    )
    page = extract(html, "https://news.example/news/2026/10/20/plan-passes")
    assert page.text_mode == MAIN
    assert (
        "Part 5. The council approved" in page.text
        and "Shop our store" not in page.text
    )


def test_pages_whose_prose_no_element_holds_get_the_whole_page():
    halves = "".join(
        f"<div class='col'><p>Column {i}. {PROSE * 3}</p></div>" for i in range(4)
    )
    html = (
        '<html><head><title>Plan passes</title><script type="application/ld+json">'
        '{"@type": "NewsArticle", "datePublished": "2026-10-20"}</script></head>'
        f"<body><h1>Plan passes</h1>{halves}</body></html>"
    )
    page = extract(html, "https://news.example/news/2026/10/20/plan-passes")
    assert page.text_mode == FULL
    assert all(f"Column {i}." in page.text for i in range(4))


def test_an_agency_byline_leaves_the_last_lines_that_share_its_common_words():
    head = (
        "<title>Plan passes</title><meta name='author' content='The Associated Press'>"
        '<script type="application/ld+json">{"@type": "NewsArticle", "datePublished": "2026-10-20"}</script>'
    )
    body = (
        f"<article><h1>Plan passes</h1><p>{PROSE * 4}</p>"
        "<p>The editor said the paper would appeal the ruling.</p>"
        "<p>more from the agency is expected next week.</p></article>"
    )
    text = extract(page(body, head), STORY_URL).text
    assert text.endswith(
        "The editor said the paper would appeal the ruling.\nmore from the agency is expected next week."
    )


def test_an_article_in_divs_without_paragraphs_is_scoped_to_its_named_container():
    sentences = "".join(f"<div>{PROSE}</div>" for _ in range(4))
    related = "".join(
        f"<div><a href='/news/s{i}'>Another story number {i} about the city</a></div>"
        for i in range(3)
    )
    body = (
        "<div class='site-top'><a href='/'>Home</a><a href='/news'>News</a><a href='/sport'>Sport</a></div>"
        f"<div class='entry-content'><h1>Council approves transit plan</h1>{sentences}"
        f"<div class='related-posts'>{related}<div class='share-bar'><a href='/s'>Share this story</a></div></div></div>"
        "<div class='sidebar'><div>Weather today is sunny with a light breeze from the west.</div></div>"
    )
    page_ = extract(page(body, ARTICLE_HEAD), STORY_URL)
    assert page_.text_mode == MAIN and "council approved" in page_.text
    for gone in ("Weather today", "Another story", "Share this story"):
        assert gone not in page_.text
