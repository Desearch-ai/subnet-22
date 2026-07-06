import unittest

from desearch.protocol import (
    ScraperStreamingSynapse,
    SearchResultItem,
    TwitterIDSearchSynapse,
    TwitterSearchSynapse,
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

    async def test_other_synapse_skipped(self):
        penalties = await self.model.calculate_penalties(
            [TwitterIDSearchSynapse(id="123", results=[{}, {}, {}])],
            [],
        )
        self.assertEqual(penalties.tolist(), [0])

    async def test_ai_search_twitter_meets_count(self):
        response = ScraperStreamingSynapse(
            prompt="x",
            count=10,
            tools=["Twitter Search"],
            miner_tweets=[{"id": str(i)} for i in range(10)],
        )
        penalties = await self.model.calculate_penalties([response])
        self.assertEqual(penalties.tolist(), [0])

    async def test_ai_search_twitter_short(self):
        response = ScraperStreamingSynapse(
            prompt="x",
            count=10,
            tools=["Twitter Search"],
            miner_tweets=[{"id": str(i)} for i in range(3)],
        )
        penalties = await self.model.calculate_penalties([response])
        self.assertAlmostEqual(penalties.tolist()[0], 0.7, places=5)

    async def test_ai_search_web_satisfied(self):
        """Web group meets count → no penalty."""
        response = ScraperStreamingSynapse(
            prompt="x",
            count=10,
            tools=["Twitter Search", "Web Search"],
            miner_tweets=[{"id": str(i)} for i in range(10)],
            search_results=[
                SearchResultItem(title=f"T{i}", link=f"https://w/{i}", snippet="s")
                for i in range(10)
            ],
        )
        penalties = await self.model.calculate_penalties([response])
        self.assertEqual(penalties.tolist(), [0])

    async def test_ai_search_web_short(self):
        """Web group below count → group-level shortfall."""
        response = ScraperStreamingSynapse(
            prompt="x",
            count=10,
            tools=["Web Search"],
            search_results=[
                SearchResultItem(title=f"T{i}", link=f"https://w/{i}", snippet="s")
                for i in range(6)
            ],
        )
        penalties = await self.model.calculate_penalties([response])
        self.assertAlmostEqual(penalties.tolist()[0], 0.4, places=5)

    async def test_ai_search_legacy_tool_not_a_group(self):
        """Legacy tools (folded to Web at the API boundary) are not scored groups."""
        response = ScraperStreamingSynapse(
            prompt="x",
            count=10,
            tools=["Reddit Search"],
            reddit_search_results=[
                SearchResultItem(title=f"R{i}", link=f"https://r/{i}", snippet="s")
                for i in range(5)
            ],
        )
        penalties = await self.model.calculate_penalties([response])
        self.assertEqual(penalties.tolist(), [0])

    async def test_ai_search_worst_group_wins(self):
        """Twitter satisfied, Web short → web dominates."""
        response = ScraperStreamingSynapse(
            prompt="x",
            count=10,
            tools=["Twitter Search", "Web Search"],
            miner_tweets=[{"id": str(i)} for i in range(10)],
            search_results=[
                SearchResultItem(title=f"W{i}", link=f"https://w/{i}", snippet="s")
                for i in range(2)
            ],
        )
        penalties = await self.model.calculate_penalties([response])
        self.assertAlmostEqual(penalties.tolist()[0], 0.8, places=5)

    async def test_ai_search_ignores_data_from_unrequested_tool(self):
        """If a tool wasn't requested, its result field isn't penalized."""
        response = ScraperStreamingSynapse(
            prompt="x",
            count=10,
            tools=["Twitter Search"],
            miner_tweets=[{"id": str(i)} for i in range(10)],
            arxiv_search_results=[
                SearchResultItem(title=f"A{i}", link=f"https://a/{i}", snippet="s")
                for i in range(2)
            ],
        )
        penalties = await self.model.calculate_penalties([response])
        self.assertEqual(penalties.tolist(), [0])

    async def test_ai_search_legacy_fields_ignored(self):
        """Twitter + Web satisfied; stray legacy result fields are ignored → 0."""
        response = ScraperStreamingSynapse(
            prompt="x",
            count=10,
            tools=["Twitter Search", "Web Search"],
            miner_tweets=[{"id": str(i)} for i in range(10)],
            search_results=[
                SearchResultItem(title=f"T{i}", link=f"https://w/{i}", snippet="s")
                for i in range(10)
            ],
            reddit_search_results=[
                SearchResultItem(title=f"R{i}", link=f"https://r/{i}", snippet="s")
                for i in range(2)
            ],
            hacker_news_search_results=[
                SearchResultItem(title=f"H{i}", link=f"https://h/{i}", snippet="s")
                for i in range(2)
            ],
        )
        penalties = await self.model.calculate_penalties([response])
        self.assertEqual(penalties.tolist(), [0])


if __name__ == "__main__":
    unittest.main()
