import unittest

from desearch.protocol import (
    ScraperStreamingSynapse,
    TwitterSearchSynapse,
)
from neurons.validators.penalty.sort_order_penalty import SortOrderPenaltyModel


def _tweet(created_at: str) -> dict:
    return {"id": "1", "created_at": created_at}


class SortOrderPenaltyTestCase(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.model = SortOrderPenaltyModel()

    async def test_latest_descending(self):
        response = TwitterSearchSynapse(
            query="x",
            sort="Latest",
            results=[
                _tweet("Tue May 15 12:00:00 +0000 2026"),
                _tweet("Mon May 10 12:00:00 +0000 2026"),
            ],
        )
        penalties = await self.model.calculate_penalties([response])
        self.assertEqual(penalties.tolist(), [0])

    async def test_latest_ascending_penalized(self):
        response = TwitterSearchSynapse(
            query="x",
            sort="Latest",
            results=[
                _tweet("Mon May 10 12:00:00 +0000 2026"),
                _tweet("Tue May 15 12:00:00 +0000 2026"),
            ],
        )
        penalties = await self.model.calculate_penalties([response])
        self.assertEqual(penalties.tolist(), [1.0])

    async def test_sort_not_latest_ignored(self):
        """Sort=Top doesn't require descending order."""
        response = TwitterSearchSynapse(
            query="x",
            sort="Top",
            results=[
                _tweet("Mon May 10 12:00:00 +0000 2026"),
                _tweet("Tue May 15 12:00:00 +0000 2026"),
            ],
        )
        penalties = await self.model.calculate_penalties([response])
        self.assertEqual(penalties.tolist(), [0])

    async def test_no_sort_field_ignored(self):
        response = TwitterSearchSynapse(
            query="x", results=[_tweet("Mon May 10 12:00:00 +0000 2026")]
        )
        penalties = await self.model.calculate_penalties([response])
        self.assertEqual(penalties.tolist(), [0])

    async def test_missing_created_at_penalized(self):
        response = TwitterSearchSynapse(
            query="x",
            sort="Latest",
            results=[{"id": "1"}],
        )
        penalties = await self.model.calculate_penalties([response])
        self.assertEqual(penalties.tolist(), [1.0])

    async def test_empty_results_passes(self):
        response = TwitterSearchSynapse(query="x", sort="Latest", results=[])
        penalties = await self.model.calculate_penalties([response])
        self.assertEqual(penalties.tolist(), [0])

    async def test_non_twitter_synapses_skipped(self):
        for response in [
            ScraperStreamingSynapse(prompt="x"),
        ]:
            penalties = await self.model.calculate_penalties([response])
            self.assertEqual(penalties.tolist(), [0])


if __name__ == "__main__":
    unittest.main()
