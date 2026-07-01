import unittest

from desearch.protocol import (
    ScraperStreamingSynapse,
    TwitterSearchSynapse,
)
from neurons.validators.penalty.domain_filter_penalty import DomainFilterPenaltyModel


def _result(link: str) -> dict:
    return {"title": "t", "link": link, "snippet": "s"}


class DomainFilterPenaltyTestCase(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.model = DomainFilterPenaltyModel()

    def _synapse(self, links, include=None, exclude=None):
        return ScraperStreamingSynapse(
            prompt="x",
            tools=["Web Search"],
            include_domains=include or [],
            exclude_domains=exclude or [],
            search_results=[_result(link) for link in links],
        )

    async def test_no_filter_skipped(self):
        response = self._synapse(["https://random.com/a"])
        penalties = await self.model.calculate_penalties([response])
        self.assertEqual(penalties.tolist(), [0])

    async def test_include_all_allowed(self):
        response = self._synapse(
            ["https://bbc.com/a", "https://news.bbc.com/b"],
            include=["bbc.com"],
        )
        penalties = await self.model.calculate_penalties([response])
        self.assertEqual(penalties.tolist(), [0])

    async def test_include_all_violating(self):
        response = self._synapse(
            ["https://cnn.com/a", "https://reuters.com/b"],
            include=["bbc.com"],
        )
        penalties = await self.model.calculate_penalties([response])
        self.assertEqual(penalties.tolist(), [1.0])

    async def test_include_half_violating(self):
        response = self._synapse(
            ["https://bbc.com/a", "https://cnn.com/b"],
            include=["bbc.com"],
        )
        penalties = await self.model.calculate_penalties([response])
        self.assertEqual(penalties.tolist(), [0.5])

    async def test_exclude_hit(self):
        response = self._synapse(
            ["https://pinterest.com/a", "https://bbc.com/b"],
            exclude=["pinterest.com"],
        )
        penalties = await self.model.calculate_penalties([response])
        self.assertEqual(penalties.tolist(), [0.5])

    async def test_exclude_subdomain_hit(self):
        response = self._synapse(
            ["https://www.pinterest.com/a"],
            exclude=["pinterest.com"],
        )
        penalties = await self.model.calculate_penalties([response])
        self.assertEqual(penalties.tolist(), [1.0])

    async def test_include_and_exclude_combined(self):
        response = self._synapse(
            [
                "https://bbc.com/a",  # ok
                "https://cnn.com/b",  # not in include
                "https://spam.bbc.com/c",  # in include but also excluded
            ],
            include=["bbc.com"],
            exclude=["spam.bbc.com"],
        )
        penalties = await self.model.calculate_penalties([response])
        self.assertAlmostEqual(penalties.tolist()[0], 2 / 3, places=5)

    async def test_no_results_skipped(self):
        response = self._synapse([], include=["bbc.com"])
        penalties = await self.model.calculate_penalties([response])
        self.assertEqual(penalties.tolist(), [0])

    async def test_messy_domain_normalized(self):
        response = self._synapse(
            ["https://cnn.com/a"],
            include=["https://BBC.com/"],
        )
        penalties = await self.model.calculate_penalties([response])
        self.assertEqual(penalties.tolist(), [1.0])

    async def test_other_synapse_types_skipped(self):
        response = TwitterSearchSynapse(query="x", results=[])
        penalties = await self.model.calculate_penalties([response])
        self.assertEqual(penalties.tolist(), [0])


if __name__ == "__main__":
    unittest.main()
