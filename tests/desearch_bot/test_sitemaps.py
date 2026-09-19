from desearch_bot.sitemaps import parse_entries


def test_fields_are_read_in_any_order_and_case():
    body = (
        b"<urlset><url><LastMod>2026-09-01</LastMod><loc> https://a.com/x </loc>"
        b"<changefreq>Daily</changefreq></url></urlset>"
    )
    kind, [entry] = parse_entries(body)
    assert (kind, entry.url, entry.lastmod, entry.changefreq) == (
        "urlset",
        "https://a.com/x",
        "2026-09-01",
        "daily",
    )


def test_the_first_value_of_a_repeated_field_wins():
    body = b"<urlset><url><loc>https://a.com/1</loc><loc>https://a.com/2</loc></url></urlset>"
    assert parse_entries(body)[1][0].url == "https://a.com/1"


def test_an_index_and_a_news_date_are_recognised():
    index = b"<sitemapindex><sitemap><loc>https://a.com/s1.xml</loc></sitemap></sitemapindex>"
    assert parse_entries(index)[0] == "index"
    news = (
        b"<urlset><url><loc>https://a.com/n</loc><news:news><news:publication_date>"
        b"2026-09-10T08:00:00Z</news:publication_date></news:news></url></urlset>"
    )
    assert parse_entries(news)[1][0].published == "2026-09-10T08:00:00Z"
