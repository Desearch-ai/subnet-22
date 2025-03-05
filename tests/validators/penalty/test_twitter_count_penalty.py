import unittest
from neurons.validators.penalty.twitter_count_penalty import TwitterCountPenaltyModel
from datura.protocol import TwitterSearchSynapse, TwitterIDSearchSynapse
import torch


class SummaryRulePenaltyTestCase(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.model = TwitterCountPenaltyModel()

    async def test_calculate_penalties_right_count(self):
        penalties = await self.model.calculate_penalties(
            [
                TwitterSearchSynapse(
                    query="What is blockchain?", count=3, results=[{}, {}, {}]
                )
            ],
            [],
        )
        self.assertEqual(penalties, torch.tensor([0]))

    async def test_calculate_penalties_not_enough_results(self):
        penalties = await self.model.calculate_penalties(
            [TwitterSearchSynapse(query="What is blockchain?", count=4, results=[{}])],
            [],
        )
        self.assertEqual(penalties, torch.tensor([0.75]))

    async def test_calculate_penalties_more_results(self):
        penalties = await self.model.calculate_penalties(
            [
                TwitterSearchSynapse(
                    query="What is blockchain?", count=2, results=[{}, {}, {}]
                )
            ],
            [],
        )
        self.assertEqual(penalties, torch.tensor([0]))

    async def test_calculate_penalties_other_synapse(self):
        penalties = await self.model.calculate_penalties(
            [TwitterIDSearchSynapse(id="123", results=[{}, {}, {}])],
            [],
        )
        self.assertEqual(penalties, torch.tensor([0]))


if __name__ == "__main__":
    unittest.main()
