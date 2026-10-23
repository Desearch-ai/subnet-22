"""Shared page fixtures for the extraction tests."""

PROSE = "The council approved the new transit plan after a long public debate about fares and routes. "


def page(body: str, head: str = "<title>City News</title>") -> str:
    return f"<html lang='en'><head>{head}</head><body>{body}</body></html>"


ARTICLE_HEAD = (
    "<title>Council approves transit plan</title>"
    '<script type="application/ld+json">{"@type": "NewsArticle", "datePublished": "2026-10-20T09:30:00+02:00",'
    ' "headline": "Council approves transit plan"}</script>'
)


ARTICLE = page(
    "<nav><a href='/'>Home</a><a href='/news'>News</a><a href='/sport'>Sport</a></nav>"
    "<main><article><h1>Council approves transit plan</h1>"
    f"<p>{PROSE * 3}</p><p>{PROSE * 3}</p>"
    "<aside class='related'><a href='/a'>Other story one</a><a href='/b'>Other story two</a><a href='/c'>Three</a></aside>"
    f"<p>{PROSE * 2}</p></article></main>"
    "<footer><a href='/privacy'>Privacy</a><a href='/terms'>Terms</a><a href='/about'>About</a></footer>",
    ARTICLE_HEAD,
)


STORY_URL = "https://news.example/2026/10/20/council-approves-transit-plan"


def story_page(body: str, before: str = "") -> str:
    return page(f"{before}<article>{body}</article>", ARTICLE_HEAD)
