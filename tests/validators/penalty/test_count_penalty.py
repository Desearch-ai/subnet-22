import unittest

from desearch.protocol import (
    ScraperStreamingSynapse,
    SearchResultItem,
    TwitterIDSearchSynapse,
    TwitterSearchSynapse,
    WebSearchSynapse,
)
from neurons.validators.penalty.count_penalty import CountPenaltyModel


class CountPenaltyTestCase(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.model = CountPenaltyModel()

    async def test_twitter_right_count(self):
        penalties = await self.model.calculate_penalties(
            [
                TwitterSearchSynapse(
                    query="What is blockchain?", count=3, results=[{}, {}, {}]
                )
            ],
            [],
        )
        self.assertEqual(penalties.tolist(), [0])

    async def test_twitter_not_enough_results(self):
        penalties = await self.model.calculate_penalties(
            [TwitterSearchSynapse(query="What is blockchain?", count=4, results=[{}])],
            [],
        )
        self.assertAlmostEqual(penalties.tolist()[0], 0.75, places=5)

    async def test_twitter_more_results(self):
        penalties = await self.model.calculate_penalties(
            [
                TwitterSearchSynapse(
                    query="What is blockchain?", count=2, results=[{}, {}, {}]
                )
            ],
            [],
        )
        self.assertEqual(penalties.tolist(), [0])

    async def test_web_right_count(self):
        penalties = await self.model.calculate_penalties(
            [
                WebSearchSynapse(
                    query="What is blockchain?",
                    num=10,
                    results=[{} for _ in range(10)],
                )
            ],
            [],
        )
        self.assertEqual(penalties.tolist(), [0])

    async def test_web_not_enough_results(self):
        penalties = await self.model.calculate_penalties(
            [
                WebSearchSynapse(
                    query="What is blockchain?", num=10, results=[{}, {}, {}]
                )
            ],
            [],
        )
        self.assertAlmostEqual(penalties.tolist()[0], 0.7, places=5)

    async def test_web_zero_results(self):
        penalties = await self.model.calculate_penalties(
            [WebSearchSynapse(query="What is blockchain?", num=10, results=[])],
            [],
        )
        self.assertEqual(penalties.tolist(), [1.0])

    async def test_other_synapse_skipped(self):
        penalties = await self.model.calculate_penalties(
            [TwitterIDSearchSynapse(id="123", results=[{}, {}, {}])],
            [],
        )
        self.assertEqual(penalties.tolist(), [0])

    async def test_ai_search_miner_tweets_meets_count(self):
        response = ScraperStreamingSynapse(
            prompt="x",
            count=10,
            miner_tweets=[{"id": str(i)} for i in range(10)],
        )
        penalties = await self.model.calculate_penalties([response])
        self.assertEqual(penalties.tolist(), [0])

    async def test_ai_search_miner_tweets_short(self):
        response = ScraperStreamingSynapse(
            prompt="x",
            count=10,
            miner_tweets=[{"id": str(i)} for i in range(3)],
        )
        penalties = await self.model.calculate_penalties([response])
        self.assertAlmostEqual(penalties.tolist()[0], 0.7, places=5)

    async def test_ai_search_worst_shortfall_across_fields(self):
        """When multiple sources are populated, worst shortfall wins."""
        response = ScraperStreamingSynapse(
            prompt="x",
            count=10,
            miner_tweets=[{"id": str(i)} for i in range(10)],
            search_results=[
                SearchResultItem(title=f"T{i}", link=f"https://a/{i}", snippet="s")
                for i in range(4)
            ],
        )
        penalties = await self.model.calculate_penalties([response])
        self.assertAlmostEqual(penalties.tolist()[0], 0.6, places=5)

    async def test_ai_search_empty_field_skipped(self):
        """An entirely-empty result field doesn't count against the miner —
        the miner didn't claim that source."""
        response = ScraperStreamingSynapse(
            prompt="x",
            count=10,
            miner_tweets=[{"id": str(i)} for i in range(10)],
        )
        penalties = await self.model.calculate_penalties([response])
        self.assertEqual(penalties.tolist(), [0])


if __name__ == "__main__":
    unittest.main()
