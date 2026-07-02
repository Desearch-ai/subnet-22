import unittest

from desearch.protocol import ScraperStreamingSynapse, SearchResultItem
from neurons.validators.reward.search_content_relevance import (
    WebSearchContentRelevanceModel,
)
from neurons.validators.reward.twitter_basic_search_content_relevance import (
    TwitterBasicSearchContentRelevanceModel,
)


class MinerLinkMetadataTestCase(unittest.TestCase):
    def setUp(self):
        self.model = WebSearchContentRelevanceModel(None, llm_reward=None, neuron=None)

    def test_handles_search_result_item_models(self):
        response = ScraperStreamingSynapse(
            prompt="q",
            search_results=[
                SearchResultItem(
                    title="t",
                    link="https://example.com/a",
                    snippet="s",
                    highlights=["h1"],
                    text="body",
                )
            ],
        )

        meta = self.model._miner_link_metadata(response)

        self.assertEqual(len(meta), 1)
        entry = next(iter(meta.values()))
        self.assertEqual(entry["highlights"], ["h1"])
        self.assertEqual(entry["text"], "body")

    def test_handles_dict_items_and_missing_link(self):
        response = ScraperStreamingSynapse(prompt="q")
        response.search_results = [
            {"title": "t", "link": "https://example.com/a", "snippet": "s"},
            {"title": "t2", "link": "", "snippet": "s2"},
        ]

        meta = self.model._miner_link_metadata(response)

        self.assertEqual(len(meta), 1)


class CompareMediaTestCase(unittest.TestCase):
    def setUp(self):
        self.model = TwitterBasicSearchContentRelevanceModel(None, neuron=None)

    def test_none_media_treated_as_empty(self):
        self.assertTrue(self.model.compare_media(None, None))
        self.assertTrue(self.model.compare_media(None, []))
        self.assertTrue(self.model.compare_media([], None))

    def test_none_vs_media_mismatch(self):
        media = [{"type": "photo", "media_url": "https://example.com/i.jpg"}]
        self.assertFalse(self.model.compare_media(None, media))
        self.assertFalse(self.model.compare_media(media, None))

    def test_matching_media(self):
        media = [{"type": "photo", "media_url": "https://example.com/i.jpg"}]
        self.assertTrue(self.model.compare_media(media, list(media)))


if __name__ == "__main__":
    unittest.main()
