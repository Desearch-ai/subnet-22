import unittest
from types import SimpleNamespace

from neurons.validators.reward.search_content_relevance import (
    WebSearchContentRelevanceModel,
)


class WebDateGateTestCase(unittest.TestCase):
    def setUp(self):
        self.model = WebSearchContentRelevanceModel.__new__(
            WebSearchContentRelevanceModel
        )
        self.response = SimpleNamespace(
            start_date="2026-01-01T00:00:00Z",
            end_date="2026-01-31T00:00:00Z",
        )

    def test_out_of_window_is_blocked(self):
        link = {"published_date": "2025-06-01T12:00:00Z"}
        self.assertTrue(self.model._web_date_blocks_link(self.response, link))

    def test_in_window_is_kept(self):
        link = {"published_date": "2026-01-15T12:00:00Z"}
        self.assertFalse(self.model._web_date_blocks_link(self.response, link))

    def test_undated_is_kept(self):
        link = {"published_date": ""}
        self.assertFalse(self.model._web_date_blocks_link(self.response, link))


class BodyFetchMetadataTestCase(unittest.TestCase):
    def test_extract_meta_returns_published_date_and_author(self):
        import neurons.validators.apify.body_fetch as bf

        class FakeMeta:
            title = "Headline"
            date = "2026-01-15"
            author = "Jane Doe"

        class FakeTrafilatura:
            @staticmethod
            def extract_metadata(html, default_url=None):
                return FakeMeta()

        import sys

        original = sys.modules.get("trafilatura")
        sys.modules["trafilatura"] = FakeTrafilatura()
        try:
            title, date, author = bf._extract_meta("<html></html>", "http://x.test")
        finally:
            if original is not None:
                sys.modules["trafilatura"] = original
            else:
                del sys.modules["trafilatura"]

        self.assertEqual(title, "Headline")
        self.assertEqual(date, "2026-01-15")
        self.assertEqual(author, "Jane Doe")


if __name__ == "__main__":
    unittest.main()
