import unittest

from desearch.protocol import (
    ScraperStreamingSynapse,
    TwitterIDSearchSynapse,
    TwitterSearchSynapse,
)
from neurons.validators.penalty.date_range_penalty import DateRangePenaltyModel


def _tweet(created_at: str) -> dict:
    return {"id": "1", "created_at": created_at}


START = "2026-05-01T00:00:00Z"
END = "2026-05-31T23:59:59Z"


class DateRangePenaltyTestCase(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.model = DateRangePenaltyModel()

    async def test_twitter_all_in_range(self):
        response = TwitterSearchSynapse(
            query="x",
            start_date=START,
            end_date=END,
            results=[
                _tweet("Mon May 10 12:00:00 +0000 2026"),
                _tweet("Tue May 15 12:00:00 +0000 2026"),
            ],
        )
        penalties = await self.model.calculate_penalties([response])
        self.assertEqual(penalties.tolist(), [0])

    async def test_twitter_all_out_of_range(self):
        response = TwitterSearchSynapse(
            query="x",
            start_date=START,
            end_date=END,
            results=[
                _tweet("Mon Apr 10 12:00:00 +0000 2026"),
                _tweet("Tue Jun 15 12:00:00 +0000 2026"),
            ],
        )
        penalties = await self.model.calculate_penalties([response])
        self.assertEqual(penalties.tolist(), [1.0])

    async def test_twitter_half_out_of_range(self):
        response = TwitterSearchSynapse(
            query="x",
            start_date=START,
            end_date=END,
            results=[
                _tweet("Mon May 10 12:00:00 +0000 2026"),  # in
                _tweet("Tue Jun 15 12:00:00 +0000 2026"),  # out
            ],
        )
        penalties = await self.model.calculate_penalties([response])
        self.assertEqual(penalties.tolist(), [0.5])

    async def test_twitter_no_bounds_skipped(self):
        response = TwitterSearchSynapse(
            query="x",
            results=[_tweet("Mon May 10 12:00:00 +0000 2026")],
        )
        penalties = await self.model.calculate_penalties([response])
        self.assertEqual(penalties.tolist(), [0])

    async def test_twitter_missing_created_at_skipped(self):
        """Tweets without created_at don't count toward the denominator."""
        response = TwitterSearchSynapse(
            query="x",
            start_date=START,
            end_date=END,
            results=[
                {"id": "1"},
                _tweet("Mon May 10 12:00:00 +0000 2026"),  # in range
            ],
        )
        penalties = await self.model.calculate_penalties([response])
        self.assertEqual(penalties.tolist(), [0])

    async def test_twitter_empty_results(self):
        response = TwitterSearchSynapse(
            query="x",
            start_date=START,
            end_date=END,
            results=[],
        )
        penalties = await self.model.calculate_penalties([response])
        self.assertEqual(penalties.tolist(), [0])

    async def test_ai_search_uses_miner_tweets(self):
        response = ScraperStreamingSynapse(
            prompt="x",
            start_date=START,
            end_date=END,
            miner_tweets=[_tweet("Mon Apr 10 12:00:00 +0000 2026")],  # out
        )
        penalties = await self.model.calculate_penalties([response])
        self.assertEqual(penalties.tolist(), [1.0])

    async def test_other_synapse_types_skipped(self):
        for response in [
            TwitterIDSearchSynapse(id="1"),
        ]:
            penalties = await self.model.calculate_penalties([response])
            self.assertEqual(penalties.tolist(), [0])


if __name__ == "__main__":
    unittest.main()
