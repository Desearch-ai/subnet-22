from desearch.extraction import FULL, extract
from desearch.extraction.text.page import MAX_LISTING_CHARS
from tests.extraction.pages import PROSE, STORY_URL, page, story_page


def test_other_pages_get_their_full_content_without_chrome():
    html = page(
        "<header class='site-header'><a href='/'>Home</a><a href='/pricing'>Pricing</a><a href='/docs'>Docs</a></header>"
        "<div id='cookie-banner'>We use cookies to improve your experience</div>"
        "<h1>Build faster</h1><p>Acme makes tools for builders who ship every day.</p>"
        "<h2>FAQ</h2><button>How long is shipping?</button><div aria-hidden='true'><p>Two days in the US.</p></div>"
        "<dialog><p>Sign up for our newsletter</p></dialog>",
        "<title>Acme Tools</title>",
    )
    got = extract(html, "https://acme.example/")
    assert (got.page_type, got.text_mode) == ("home", FULL)
    assert got.text.splitlines()[0] == "Acme Tools"
    assert (
        "# Build faster" in got.text
        and "How long is shipping?" in got.text
        and "Two days in the US." in got.text
    )
    assert (
        "cookies" not in got.text
        and "Pricing" not in got.text
        and "newsletter" not in got.text
    )


def test_drops_link_runs_skip_links_repeats_and_blobs_but_keeps_cards():
    menu = "".join(f"<li><a href='/c{i}'>Category {i}</a></li>" for i in range(10))
    cards = "".join(
        f"<div><a href='/p{i}'>Shelf model {i}</a><p>Pine shelving that grows with you.</p></div>"
        for i in range(3)
    )
    html = page(
        f"<a href='#main'>Skip to main content</a><main><ul>{menu}</ul>{cards}"
        "<p>This repeated line has more than eight words in it.</p><p>This repeated line has more than eight words in it.</p>"
        "<p>$12</p><p>$12</p><p>" + "QUJD" * 100 + "</p></main>"
    )
    text = extract(html, "https://a.example/").text
    assert "Category 4" not in text and "Skip to main content" not in text
    assert (
        "Shelf model 1" in text
        and text.count("more than eight words") == 1
        and text.count("$12") == 2
        and "QUJD" not in text
    )


def test_listings_are_capped():
    items = "".join(
        f"<h3><a href='/s{i}'>Story number {i} about the city council</a></h3><p>{PROSE}</p>"
        for i in range(400)
    )
    got = extract(
        page(f"<main>{items}</main>"), "https://news.example/category/politics/"
    )
    assert got.page_type == "listing" and len(got.text) <= MAX_LISTING_CHARS


def test_forum_threads_keep_every_post():
    head = '<title>Brake noise</title><script type="application/ld+json">{"@type": "DiscussionForumPosting"}</script>'
    posts = "".join(f"<article><p>Reply {i}: {PROSE}</p></article>" for i in range(5))
    got = extract(page(posts, head), "https://forum.example/t/brake-noise/123")
    assert (got.page_type, got.text_mode) == ("forum", FULL) and "Reply 4:" in got.text


def test_product_links_stay_and_on_site_link_boxes_go():
    body = (
        f"<h1>Council approves transit plan</h1><p>{PROSE * 4}</p>"
        "<div><a href='https://shop.example/card'>Transit card holder</a> $12</div>"
        "<p><a href='/news/older'>An older story about the council</a></p>"
    )
    text = extract(story_page(body), STORY_URL).text
    assert "Transit card holder" in text and "older story" not in text


def test_icon_words_and_media_fallbacks_are_dropped():
    html = page(
        "<p><span class='material-symbols-outlined icon'>schedule</span> 2 min read</p>"
        "<div class='field field--label-visually_hidden'><p>Notice of the council meeting.</p></div>"
        "<video src='a.mp4'>Your browser does not support video.</video>"
    )
    text = extract(html, "https://a.example/").text
    assert "schedule" not in text and "support video" not in text
    assert "2 min read" in text and "Notice of the council meeting." in text


def test_hidden_labels_count_on_whole_pages_but_not_in_articles():
    rating = "<p>Rated <span class='visually-hidden'>4.5 out of 5 stars</span></p><span role='tooltip'>Ships in 2 days</span>"
    assert (
        "4.5 out of 5 stars"
        in extract(page(f"<p>Pine shelf $49</p>{rating}"), "https://shop.example/").text
    )
    body = f"<h1>Council approves transit plan</h1><p>{PROSE * 4}</p><span class='sr-only'>list of 3 items</span>"
    assert "list of 3 items" not in extract(story_page(body), STORY_URL).text


def test_whole_pages_drop_chrome_but_keep_content():
    rows = "".join(
        f"<li><a href='/m{i}'>Menu item {i}</a><ul><li><a href='/s{i}'>Sub {i}</a></li></ul></li>"
        for i in range(5)
    )
    html = page(
        f"<div class='MobileMenu-module'><ul>{rows}</ul></div>"
        "<header class='site-header'><h1>Build faster</h1><p>Tools for builders.</p>"
        "<nav><a href='/a'>Pricing</a><a href='/b'>Docs</a><a href='/c'>Blog</a></nav></header>"
        "<div class='breadcrumbs'><a href='/'>Home</a> / <a href='/p'>Pricing</a></div>"
        "<div class='announcement-bar'>Free shipping this week</div>"
        f"<section><h2>Plans</h2><div class='plan'><h3>Basic</h3><p>$0</p><p>Unlimited users</p></div>"
        f"<div class='plan'><h3>Pro</h3><p>$29</p><p>Unlimited users</p></div><p>{PROSE * 3}</p></section>"
        "<div class='modal'><p>Join our mailing list today</p></div>"
        "<form><label>Email</label><button>Subscribe</button></form>"
        "<div hidden id='S:1'><p>Comments from readers stream in here.</p></div>"
        '<div>{"product_id": 1, "variant_id": 2}</div>'
        "<p>Write to <a href='/cdn-cgi/l/email-protection'><span class='__cf_email__' data-cfemail='422a272e2e2d02273a232f322e276c212d2f'>[email protected]</span></a> or visit the .gov site.</p>"
        "<div class='site-footer'><a href='/t'>Terms</a><a href='/p'>Privacy</a><a href='/c'>Cookies</a> © 2026 Acme</div>"
    )
    text = extract(html, "https://acme.example/pricing").text
    assert "# Build faster" in text and "Tools for builders." in text
    assert "$0" in text and "$29" in text and text.count("Unlimited users") == 2
    assert "hello@example.com" in text and "visit the .gov site" in text
    assert "Comments from readers stream in here." in text
    for junk in (
        "Menu item",
        "Docs",
        "Free shipping",
        "mailing list",
        "Subscribe",
        "full page refresh",
        "product_id",
        "Terms",
        "/ Pricing",
    ):
        assert junk not in text, junk


def test_product_pages_drop_other_products():
    others = "".join(
        f"<div class='card'><a href='/p{i}'>Other shelf {i}</a> $9</div>"
        for i in range(4)
    )
    html = page(
        f"<h1>Pine shelf</h1><p>{PROSE * 2}</p><p>$49</p><button>Add to cart</button><div class='related-products'>{others}</div>"
    )
    text = extract(html, "https://shop.example/products/pine-shelf").text
    assert "Pine shelf" in text and "$49" in text and "Other shelf" not in text


def test_whole_pages_drop_interface_controls_but_keep_their_content():
    faq = "<button aria-expanded='false'>What does the plan cost?</button><div>It is free for schools.</div>"
    cards = "".join(
        f"<div class='card'><h3>Service {i}</h3><p>{PROSE}</p><a href='/s/{i}'>Learn more ›</a></div>"
        for i in range(3)
    )
    html = (
        "<html><head><title>Acme</title></head><body>"
        f"<h1>Acme for schools</h1><p>{PROSE * 2}</p>"
        "<button class='get-started'>Get started</button><button>I already have an account</button>"
        "<div aria-live='polite'>Item 1 of 3</div><button>Monthly $12</button>"
        "<button>Keep in mind that every school gets its own private and secure space</button>"
        f"{faq}{cards}"
        "<aside aria-label='Side social links'><a href='/fb'>Facebook</a> <a href='/yt'>YouTube</a></aside>"
        "</body></html>"
    )
    text = extract(html, "https://acme.example/schools").text
    for gone in (
        "Get started",
        "I already have an account",
        "Item 1 of 3",
        "Monthly $12",
        "Learn more",
        "Facebook",
    ):
        assert gone not in text
    for kept in (
        "What does the plan cost?",
        "It is free for schools.",
        "private and secure space",
        "Service 2",
    ):
        assert kept in text


def test_whole_pages_keep_hidden_text_only_with_a_value_or_as_a_tooltip():
    html = (
        "<html><head><title>Shelf</title></head><body>"
        "<a class='visually-hidden' href='#main'>Skip to product information</a>"
        f"<h1>Pine shelf</h1><p>{PROSE * 2}</p>"
        "<div class='price'><span class='visually-hidden'>Sale price</span><span>$49.99</span></div>"
        "<a href='/care'>Care guide<span class='visually-hidden'>Opens in a new window.</span></a>"
        "<li>Express support<div role='tooltip'>Help with setup for small teams.</div></li>"
        "</body></html>"
    )
    text = extract(html, "https://shop.example/pine-shelf").text
    assert "Sale price" in text and "$49.99" in text
    assert "Help with setup for small teams." in text
    assert "Opens in a new window" not in text and "Skip to product" not in text


def test_whole_pages_drop_sign_up_boxes_and_boilerplate_lines():
    signup = (
        "<div class='box'><h3>Stay in the loop</h3><p>Weekly updates on new products.</p>"
        "<form><input type='email'><button>Subscribe</button></form></div>"
    )
    html = (
        "<html><head><title>Acme</title></head><body>"
        "<p>You are using an out of date browser. It may not display this or other websites correctly.</p>"
        "<p>A .gov website belongs to an official government organization in the United States.</p>"
        f"<h1>Acme tasks</h1><p>{PROSE * 2}</p><p>Follow up tasks with comments</p><p>Image 2 of 4</p>"
        "<ul><li id='a11y-new-window-message'>Opens in a new window.</li></ul>"
        f"{signup}<p>When you purchase through links on our site, we may earn an affiliate commission.</p>"
        "</body></html>"
    )
    text = extract(html, "https://acme.example/tasks").text
    assert "Follow up tasks with comments" in text and "council approved" in text
    for gone in (
        "out of date browser",
        ".gov website",
        "Image 2 of 4",
        "Opens in a new window",
        "Weekly updates",
        "affiliate commission",
    ):
        assert gone not in text


def test_sign_up_boxes_go_whatever_their_scripts_and_field_spelling():
    box = (
        "<div><h3>Stay in the loop</h3><p>Weekly updates on new products.</p>"
        f"<script>{'var tracking = 1;' * 40}</script>"
        "<form><input type='EMAIL'><button>Subscribe</button></form></div>"
    )
    text = extract(
        page(f"<h1>Acme</h1><p>{PROSE * 2}</p>{box}"), "https://acme.example/"
    ).text
    assert "council approved" in text and "Weekly updates" not in text


def test_a_footer_with_contact_details_keeps_them_and_loses_its_link_lists():
    menu = "".join(
        f"<li><a href='/m{i}'>Section {i}</a><ul><li><a href='/s{i}'>Page {i}</a></li></ul></li>"
        for i in range(8)
    )
    footer = (
        "<footer><p>Acme Tools, 12 Mill Road, Springfield. Call (555) 123-4567 or write to help@acme.example.</p>"
        f"<ul>{menu}</ul><div><a href='/t'>Terms</a><a href='/p'>Privacy</a><a href='/c'>Cookies</a></div></footer>"
    )
    text = extract(
        page(f"<h1>Acme</h1><p>{PROSE * 3}</p>{footer}"), "https://acme.example/"
    ).text
    assert "12 Mill Road" in text and "(555) 123-4567" in text
    assert "Section 3" not in text and "Privacy" not in text
