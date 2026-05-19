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

    async def test_ai_search_pooled_search_summary_satisfied(self):
        """Web/Wiki/YT/ArXiv pool into one SEARCH_SUMMARY group — 7 web + 3 arxiv = 10 → no penalty."""
        response = ScraperStreamingSynapse(
            prompt="x",
            count=10,
            tools=["Twitter Search", "Web Search", "Wikipedia Search", "ArXiv Search"],
            miner_tweets=[{"id": str(i)} for i in range(10)],
            search_results=[
                SearchResultItem(title=f"T{i}", link=f"https://w/{i}", snippet="s")
                for i in range(7)
            ],
            arxiv_search_results=[
                SearchResultItem(title=f"A{i}", link=f"https://a/{i}", snippet="s")
                for i in range(3)
            ],
        )
        penalties = await self.model.calculate_penalties([response])
        self.assertEqual(penalties.tolist(), [0])

    async def test_ai_search_pooled_search_summary_short(self):
        """Pooled group below count → group-level shortfall."""
        response = ScraperStreamingSynapse(
            prompt="x",
            count=10,
            tools=["Web Search", "ArXiv Search"],
            search_results=[
                SearchResultItem(title=f"T{i}", link=f"https://w/{i}", snippet="s")
                for i in range(4)
            ],
            arxiv_search_results=[
                SearchResultItem(title=f"A{i}", link=f"https://a/{i}", snippet="s")
                for i in range(2)
            ],
        )
        penalties = await self.model.calculate_penalties([response])
        self.assertAlmostEqual(penalties.tolist()[0], 0.4, places=5)

    async def test_ai_search_reddit_solo_group(self):
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
        self.assertAlmostEqual(penalties.tolist()[0], 0.5, places=5)

    async def test_ai_search_worst_group_wins(self):
        """Twitter satisfied, Reddit short → reddit dominates."""
        response = ScraperStreamingSynapse(
            prompt="x",
            count=10,
            tools=["Twitter Search", "Reddit Search"],
            miner_tweets=[{"id": str(i)} for i in range(10)],
            reddit_search_results=[
                SearchResultItem(title=f"R{i}", link=f"https://r/{i}", snippet="s")
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

    async def test_ai_search_kitchen_sink_regression(self):
        """Regression for the reported bug: 7-tool synapse with
        miner_tweets=10, reddit=10, hn=10, web=7, arxiv=3, wiki=0, yt=0.
        Pre-fix: penalty=0.70 (arxiv field). After fix: all groups satisfied → 0."""
        response = ScraperStreamingSynapse(
            prompt="x",
            count=10,
            tools=[
                "Twitter Search",
                "Web Search",
                "Wikipedia Search",
                "Youtube Search",
                "ArXiv Search",
                "Reddit Search",
                "Hacker News Search",
            ],
            miner_tweets=[{"id": str(i)} for i in range(10)],
            search_results=[
                SearchResultItem(title=f"T{i}", link=f"https://w/{i}", snippet="s")
                for i in range(7)
            ],
            arxiv_search_results=[
                SearchResultItem(title=f"A{i}", link=f"https://a/{i}", snippet="s")
                for i in range(3)
            ],
            reddit_search_results=[
                SearchResultItem(title=f"R{i}", link=f"https://r/{i}", snippet="s")
                for i in range(10)
            ],
            hacker_news_search_results=[
                SearchResultItem(title=f"H{i}", link=f"https://h/{i}", snippet="s")
                for i in range(10)
            ],
        )
        penalties = await self.model.calculate_penalties([response])
        self.assertEqual(penalties.tolist(), [0])


if __name__ == "__main__":
    unittest.main()
