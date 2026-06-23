import unittest
from neurons.validators.penalty.miner_score_penalty import MinerScorePenaltyModel
from desearch.protocol import (
    ScraperTextRole,
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
                        "1897719318743327227": 3.0,
                    },
                    {
                        "1897719318743327227": 1.5,
                    },
                    {
                        "1897719318743327227": 3.0,
                    },
                    {},
                ],
                [
                    {
                        "https://www.investopedia.com/terms/b/blockchain.asp": 1.5,
                    },
                    {
                        "https://www.investopedia.com/terms/b/blockchain.asp": 3.0,
                    },
                    {
                        "https://www.investopedia.com/terms/b/blockchain.asp": 1.5,
                    },
                    {},
                ],
            ],
        )
        self.assertEqual(penalties.tolist(), [0, 1, 1, 0])


if __name__ == "__main__":
    unittest.main()
