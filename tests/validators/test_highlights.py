import unittest

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


if __name__ == "__main__":
    unittest.main()
