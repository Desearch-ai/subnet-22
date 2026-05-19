import unittest

from desearch.protocol import (
    ScraperStreamingSynapse,
    SearchResultItem,
    TwitterSearchSynapse,
    WebSearchSynapse,
)
from neurons.validators.penalty.duplicate_results_penalty import (
    DuplicateResultsPenaltyModel,
)


class DuplicateResultsPenaltyTestCase(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.model = DuplicateResultsPenaltyModel()

    async def test_twitter_unique(self):
        response = TwitterSearchSynapse(
            query="x",
            results=[{"id": "1"}, {"id": "2"}, {"id": "3"}],
        )
        penalties = await self.model.calculate_penalties([response])
        self.assertEqual(penalties.tolist(), [0])

    async def test_twitter_duplicate_ids(self):
        response = TwitterSearchSynapse(
            query="x",
            results=[{"id": "1"}, {"id": "2"}, {"id": "1"}],
        )
        penalties = await self.model.calculate_penalties([response])
        self.assertEqual(penalties.tolist(), [1.0])

    async def test_web_duplicate_links(self):
        response = WebSearchSynapse(
            query="x",
            num=10,
            results=[
                {"link": "https://a"},
                {"link": "https://b"},
                {"link": "https://a"},
            ],
        )
        penalties = await self.model.calculate_penalties([response])
        self.assertEqual(penalties.tolist(), [1.0])

    async def test_web_unique_links(self):
        response = WebSearchSynapse(
            query="x",
            num=10,
            results=[{"link": "https://a"}, {"link": "https://b"}],
        )
        penalties = await self.model.calculate_penalties([response])
        self.assertEqual(penalties.tolist(), [0])

    async def test_ai_dup_in_miner_tweets(self):
        response = ScraperStreamingSynapse(
            prompt="x",
            miner_tweets=[{"id": "1"}, {"id": "1"}],
        )
        penalties = await self.model.calculate_penalties([response])
        self.assertEqual(penalties.tolist(), [1.0])

    async def test_ai_dup_in_search_results(self):
        response = ScraperStreamingSynapse(
            prompt="x",
            search_results=[
                SearchResultItem(title="T1", link="https://a", snippet="s1"),
                SearchResultItem(title="T2", link="https://a", snippet="s2"),
            ],
        )
        penalties = await self.model.calculate_penalties([response])
        self.assertEqual(penalties.tolist(), [1.0])

    async def test_ai_unique_across_groups(self):
        response = ScraperStreamingSynapse(
            prompt="x",
            miner_tweets=[{"id": "1"}, {"id": "2"}],
            search_results=[
                SearchResultItem(title="T1", link="https://a", snippet="s1"),
                SearchResultItem(title="T2", link="https://b", snippet="s2"),
            ],
        )
        penalties = await self.model.calculate_penalties([response])
        self.assertEqual(penalties.tolist(), [0])

    async def test_empty(self):
        response = TwitterSearchSynapse(query="x", results=[])
        penalties = await self.model.calculate_penalties([response])
        self.assertEqual(penalties.tolist(), [0])


if __name__ == "__main__":
    unittest.main()
