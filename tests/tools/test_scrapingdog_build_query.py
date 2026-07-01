import unittest

from desearch.tools.search.scrapingdog_google_search import ScrapingDogGoogleSearch


class ScrapingDogBuildQueryTestCase(unittest.TestCase):
    def test_plain_query(self):
        search = ScrapingDogGoogleSearch()
        self.assertEqual(search.build_query("ai news"), "ai news")

    def test_single_include_domain(self):
        search = ScrapingDogGoogleSearch(include_domains=["bbc.com"])
        self.assertEqual(search.build_query("ai news"), "ai news site:bbc.com")

    def test_multiple_include_domains_grouped(self):
        search = ScrapingDogGoogleSearch(include_domains=["bbc.com", "reuters.com"])
        self.assertEqual(
            search.build_query("ai news"),
            "ai news (site:bbc.com OR site:reuters.com)",
        )

    def test_exclude_domains(self):
        search = ScrapingDogGoogleSearch(exclude_domains=["pinterest.com", "quora.com"])
        self.assertEqual(
            search.build_query("ai news"),
            "ai news -site:pinterest.com -site:quora.com",
        )

    def test_include_and_exclude(self):
        search = ScrapingDogGoogleSearch(
            include_domains=["bbc.com"], exclude_domains=["pinterest.com"]
        )
        self.assertEqual(
            search.build_query("ai news"),
            "ai news site:bbc.com -site:pinterest.com",
        )

    def test_domains_normalized_and_deduped(self):
        search = ScrapingDogGoogleSearch(
            include_domains=["https://BBC.com/", "bbc.com", ""]
        )
        self.assertEqual(search.build_query("ai news"), "ai news site:bbc.com")


if __name__ == "__main__":
    unittest.main()
