from desearch.extraction import extract
from tests.extraction.pages import PROSE, page


def test_markdown_blocks():
    html = page(
        "<h2>Steps</h2><ol><li>Mix the flour<ul><li>Sift it first</li></ul></li><li>Bake for an hour</li></ol>"
        "<table><caption>Fares</caption><tr><th>Zone</th><th>Price</th><th>Note</th></tr>"
        "<tr><td rowspan='2'>A</td><td>$2</td><td>peak</td></tr><tr><td>$1</td><td>off-peak</td></tr>"
        "<tr><td colspan='2'>Both zones</td><td>$3</td></tr></table>"
        "<table><tr><td>Single column one</td></tr><tr><td>Single column two</td></tr></table>"
        "<pre>def total(x):\n    return x  * 2</pre>"
        "<blockquote><p>Transit is a public good.</p><footer>Mayor Jane Doe</footer></blockquote>"
        "<figure><img src='a.png' alt='photo'><figcaption>The new tram at dawn</figcaption></figure>"
        "<p>Ridership rose<sup><a href='#cite-1'>[1]</a></sup> last year.</p>"
    )
    text = extract(html, "https://a.example/").text
    assert (
        "## Steps" in text
        and "1. Mix the flour" in text
        and "  - Sift it first" in text
        and "2. Bake for an hour" in text
    )
    assert "Fares" in text and "| Zone | Price | Note |\n|---|---|---|" in text
    assert (
        "| A | $2 | peak |" in text
        and "| A | $1 | off-peak |" in text
        and "| Both zones | Both zones | $3 |" in text
    )
    assert "Single column one\nSingle column two" in text
    assert "```\ndef total(x):\n    return x  * 2\n```" in text
    assert "> Transit is a public good." in text and "> Mayor Jane Doe" in text
    assert "The new tram at dawn" in text and "photo" not in text
    assert "Ridership rose last year." in text


def test_inline_spacing_and_drop_caps():
    html = page(
        "<p><span class='dropcap'>T</span>he tram runs <a href='/x'>every</a><time>10:15</time> daily.</p>"
        "<p><span class='flex gap'>01</span><span>Add your first product</span></p>"
    )
    text = extract(html, "https://a.example/").text
    assert (
        "The tram runs every 10:15 daily." in text
        and "01 Add your first product" in text
    )


def test_headings_skip_fenced_code():
    html = page(
        "<h2>Install</h2><pre># install it\npip install x</pre><h3>Use</h3><p>Run it.</p>"
    )
    assert extract(html, "https://a.example/").headings == ["Install", "Use"]


def test_side_by_side_items_get_a_space_but_split_words_do_not():
    html = page(
        "<div><span>Alex Meakin</span><span>in Bucklebury</span></div>"
        "<p><span>O</span><span>ur plan</span></p><p>Oct 21 2026<span></span>10:40 AM</p>"
    )
    text = extract(html, "https://a.example/").text
    assert (
        "Alex Meakin in Bucklebury" in text
        and "Our plan" in text
        and "Oct 21 2026 10:40 AM" in text
    )


def test_a_cell_spanning_the_row_is_written_once():
    html = page(
        "<table><tr><th>Zone</th><th>Price</th></tr><tr><td>A</td><td>$2</td></tr>"
        "<tr><td colspan='2'>Prices from October</td></tr></table>"
    )
    assert "| Prices from October |  |" in extract(html, "https://a.example/").text


def test_link_padding_never_lands_inside_quotes():
    html = page(
        "<p>showing Holmes “<a href='/x'>Who’s on first</a>,” and <a href='/t'>Trump</a>’s plan, "
        "the ’90s and rock ’n’ roll</p>"
    )
    text = extract(html, "https://a.example/").text
    assert (
        "Holmes “Who’s on first,” and Trump’s plan, the ’90s and rock ’n’ roll" in text
    )


def test_code_keeps_its_own_spacing():
    html = page(
        "<pre><code><span>function</span><span> compact</span>(a) {}\n<span>agency_id</span>,<span>name</span></code></pre>"
    )
    text = extract(html, "https://a.example/").text
    assert "function compact(a) {}" in text and "agency_id,name" in text


def test_lists_nested_inside_list_items_and_card_items():
    html = page(
        "<ul><li><a href='/x'>Shop</a><div><ul><li>Tinted lip balm</li><li>Vitamin serum</li></ul></div></li>"
        "<li><h3>Pro plan</h3><p>For growing teams.</p></li></ul>"
    )
    text = extract(html, "https://shop.example/").text
    assert "  - Tinted lip balm" in text and "  - Vitamin serum" in text
    assert "- Pro plan\nFor growing teams." in text


def test_ordered_lists_keep_their_numbers_and_nested_lists_their_item():
    html = (
        "<html><head><title>Steps</title></head><body><h1>Steps</h1>"
        f"<p>{PROSE * 2}</p>"
        "<ol start='4'><li>Fourth</li><li>Fifth<ul><li>Detail</li></ul></li></ol>"
        "<ol reversed><li>Three</li><li>Two</li><li>One</li></ol>"
        "<ol><li value='10'>Ten</li><li>Eleven<ol><li>Eleven a</li></ol></li></ol>"
        "</body></html>"
    )
    lines = extract(html, "https://guide.example/steps").text.splitlines()
    for expected in (
        "4. Fourth",
        "5. Fifth",
        "   - Detail",
        "3. Three",
        "1. One",
        "10. Ten",
        "11. Eleven",
        "    1. Eleven a",
    ):
        assert expected in lines


def test_definition_lists_pair_each_term_with_its_definition():
    html = (
        "<html><head><title>Shelf</title></head><body><h1>Pine shelf</h1>"
        f"<p>{PROSE * 2}</p><dl><dt>Width</dt><dd>80 cm</dd><div><dt>Material</dt><dd>Solid pine</dd></div>"
        "<dt>Care</dt><dd><p>Wipe with a damp cloth.</p><p>Do not use solvents.</p></dd></dl>"
        "</body></html>"
    )
    lines = extract(html, "https://shop.example/pine-shelf").text.splitlines()
    assert "Width: 80 cm" in lines and "Material: Solid pine" in lines
    assert lines[lines.index("Care") + 1 :][:2] == [
        "Wipe with a damp cloth.",
        "Do not use solvents.",
    ]
