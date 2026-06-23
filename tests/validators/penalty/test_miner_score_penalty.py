import unittest
from neurons.validators.penalty.miner_score_penalty import MinerScorePenaltyModel
from desearch.protocol import (
    ScraperStreamingSynapse,
    ContextualRelevance,
)


class MinerScorePenaltyTestCase(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.model = MinerScorePenaltyModel()

    async def test_calculate_penalties(self):
        penalties = await self.model.calculate_penalties(
            [
                ScraperStreamingSynapse(
                    prompt="blockchain",
                    tools=["Web Search"],
                    miner_link_scores={
                        "https://www.investopedia.com/terms/b/blockchain.asp": ContextualRelevance.MEDIUM,
                        "1897719318743327227": ContextualRelevance.HIGH,
                    },
                ),
                ScraperStreamingSynapse(
                    prompt="What is crypto?",
                    tools=["Web Search"],
                    miner_link_scores={
                        "https://www.investopedia.com/terms/b/blockchain.asp": ContextualRelevance.MEDIUM,
                        "1897719318743327227": ContextualRelevance.HIGH,
                    },
                ),
                ScraperStreamingSynapse(
                    prompt="What is blockchain?",
                    tools=["Web Search"],
                    miner_link_scores={},
                ),
                ScraperStreamingSynapse(
                    prompt="What is blockchain?",
                    tools=["Web Search"],
                    miner_link_scores={
                        "https://www.investopedia.com/terms/b/blockchain.asp": ContextualRelevance.MEDIUM,
                        "1897719318743327227": ContextualRelevance.HIGH,
                    },
                ),
            ],
            [
                [
                    {
                        "1897719318743327227": "HIGH",
                    },
                    {
                        "1897719318743327227": "MEDIUM",
                    },
                    {
                        "1897719318743327227": "HIGH",
                    },
                    {},
                ],
                [
                    {
                        "https://www.investopedia.com/terms/b/blockchain.asp": "MEDIUM",
                    },
                    {
                        "https://www.investopedia.com/terms/b/blockchain.asp": "HIGH",
                    },
                    {
                        "https://www.investopedia.com/terms/b/blockchain.asp": "MEDIUM",
                    },
                    {},
                ],
            ],
        )
        self.assertEqual(penalties.tolist(), [0, 1, 1, 0])


class RelevanceValueTestCase(unittest.TestCase):
    def test_normalizes_enum_and_string(self):
        from neurons.validators.penalty.miner_score_penalty import _relevance_value

        self.assertEqual(_relevance_value(ContextualRelevance.MEDIUM), "MEDIUM")
        self.assertEqual(_relevance_value("MEDIUM"), "MEDIUM")
        self.assertIsNone(_relevance_value(None))


if __name__ == "__main__":
    unittest.main()
