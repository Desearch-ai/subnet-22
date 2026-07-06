import unittest

from desearch.protocol import (
    ScraperStreamingSynapse,
    SearchResultItem,
    TwitterSearchSynapse,
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

    async def test_twitter_duplicate_urls(self):
        response = TwitterSearchSynapse(
            query="x",
            results=[
                {"id": "1", "url": "https://x.com/a/status/1"},
                {"id": "2", "url": "https://x.com/a/status/1"},
            ],
        )
        penalties = await self.model.calculate_penalties([response])
        self.assertEqual(penalties.tolist(), [1.0])

    async def test_twitter_duplicate_texts(self):
        response = TwitterSearchSynapse(
            query="x",
            results=[
                {"id": "1", "url": "https://x.com/a/status/1", "text": "Same tweet"},
                {"id": "2", "url": "https://x.com/b/status/2", "text": " same  tweet "},
            ],
        )
        penalties = await self.model.calculate_penalties([response])
        self.assertEqual(penalties.tolist(), [1.0])

    async def test_single_duplicate_zeroes_response_multiplier(self):
        response = ScraperStreamingSynapse(
            prompt="x",
            search_results=[
                SearchResultItem(title="T1", link="https://a", snippet="s1"),
                SearchResultItem(title="T2", link="https://b", snippet="s2"),
                SearchResultItem(title="T3", link="https://a", snippet="s3"),
            ],
        )
        _, _, applied = await self.model.apply_penalties([response], uids=[0])
        self.assertEqual(applied.tolist(), [0.0])

    async def test_ai_dup_in_miner_tweets(self):
        response = ScraperStreamingSynapse(
            prompt="x",
            miner_tweets=[{"id": "1"}, {"id": "1"}],
        )
        penalties = await self.model.calculate_penalties([response])
        self.assertEqual(penalties.tolist(), [1.0])

    async def test_ai_dup_in_miner_tweet_urls(self):
        response = ScraperStreamingSynapse(
            prompt="x",
            miner_tweets=[
                {"id": "1", "url": "https://x.com/a/status/1"},
                {"id": "2", "url": "https://x.com/a/status/1"},
            ],
        )
        penalties = await self.model.calculate_penalties([response])
        self.assertEqual(penalties.tolist(), [1.0])

    async def test_ai_dup_in_miner_tweet_texts(self):
        response = ScraperStreamingSynapse(
            prompt="x",
            miner_tweets=[
                {"id": "1", "url": "https://x.com/a/status/1", "text": "Same tweet"},
                {"id": "2", "url": "https://x.com/b/status/2", "text": "same tweet"},
            ],
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

    async def test_www_tracking_param_variant_is_duplicate(self):
        response = ScraperStreamingSynapse(
            prompt="x",
            search_results=[
                SearchResultItem(title="T1", link="https://a/page", snippet="s1"),
                SearchResultItem(
                    title="T2",
                    link="https://www.a/page/?utm_source=x&fbclid=y",
                    snippet="s2",
                ),
            ],
        )
        penalties = await self.model.calculate_penalties([response])
        self.assertEqual(penalties.tolist(), [1.0])

    async def test_content_param_variant_is_distinct(self):
        response = ScraperStreamingSynapse(
            prompt="x",
            search_results=[
                SearchResultItem(
                    title="T1", link="https://www.youtube.com/watch?v=AAA", snippet="s1"
                ),
                SearchResultItem(
                    title="T2", link="https://www.youtube.com/watch?v=BBB", snippet="s2"
                ),
            ],
        )
        penalties = await self.model.calculate_penalties([response])
        self.assertEqual(penalties.tolist(), [0])

    async def test_ai_search_results_tracking_param_variant_is_duplicate(self):
        response = ScraperStreamingSynapse(
            prompt="x",
            search_results=[
                SearchResultItem(title="T1", link="https://a/p", snippet="s1"),
                SearchResultItem(
                    title="T2", link="https://a/p?utm_campaign=z", snippet="s2"
                ),
            ],
        )
        penalties = await self.model.calculate_penalties([response])
        self.assertEqual(penalties.tolist(), [1.0])

    async def test_empty(self):
        response = TwitterSearchSynapse(query="x", results=[])
        penalties = await self.model.calculate_penalties([response])
        self.assertEqual(penalties.tolist(), [0])


if __name__ == "__main__":
    unittest.main()
