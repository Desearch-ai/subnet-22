import unittest

from neurons.validators.reward.search_content_relevance import link_meets_evidence
from neurons.validators.utils.source_bodies import highlight_subset_of_body


class HighlightSubsetTestCase(unittest.TestCase):
    def test_returns_only_present_highlights(self):
        body = "The Eclipse release shipped on March 3rd with a new search engine."
        highlights = [
            "the eclipse   RELEASE shipped",
            "a NEW search engine",
            "this sentence was never on the page",
        ]

        verified = highlight_subset_of_body(highlights, body)

        self.assertEqual(
            verified, ["the eclipse   RELEASE shipped", "a NEW search engine"]
        )

    def test_handles_entities_and_whitespace(self):
        body = "Profits rose 5% &amp; revenue doubled."
        self.assertEqual(
            highlight_subset_of_body(["profits rose 5%   & revenue"], body),
            ["profits rose 5%   & revenue"],
        )

    def test_empty_body_returns_nothing(self):
        self.assertEqual(highlight_subset_of_body(["anything"], ""), [])

    def test_non_latin_scripts_are_matched(self):
        body = "市長宣布造價兩億元的新大橋將於明年開放通車。"
        self.assertEqual(
            highlight_subset_of_body(["造價兩億元的新大橋將於明年開放"], body),
            ["造價兩億元的新大橋將於明年開放"],
        )
        self.assertEqual(highlight_subset_of_body(["這座大橋已被拆除"], body), [])


class LinkEvidenceGateTestCase(unittest.TestCase):
    body = "The Federal Reserve held rates steady at its June meeting. Officials projected one cut."
    highlights = ["The Federal Reserve held rates steady at its June meeting"]

    def test_passes_with_highlights_and_matching_text(self):
        self.assertTrue(link_meets_evidence(self.highlights, self.body, self.body))

    def test_fails_without_highlights(self):
        self.assertFalse(link_meets_evidence([], self.body, self.body))

    def test_fails_without_page_text(self):
        self.assertFalse(link_meets_evidence(self.highlights, "", self.body))

    def test_fails_when_highlights_not_on_fetched_page(self):
        self.assertFalse(
            link_meets_evidence(self.highlights, self.body, "unrelated page text")
        )

    def test_fails_when_text_inconsistent_with_highlights(self):
        self.assertFalse(
            link_meets_evidence(self.highlights, "some other page text", self.body)
        )

    def test_passes_when_highlights_in_document_order(self):
        ordered = [
            "The Federal Reserve held rates steady at its June meeting",
            "Officials projected one cut",
        ]
        self.assertTrue(link_meets_evidence(ordered, self.body, self.body))

    def test_fails_when_highlights_out_of_order(self):
        reordered = [
            "Officials projected one cut",
            "The Federal Reserve held rates steady at its June meeting",
        ]
        self.assertFalse(link_meets_evidence(reordered, self.body, self.body))


if __name__ == "__main__":
    unittest.main()
