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
            [],
            [
                [
                    {
                        "1897719318743327227": 9.0,
                    },
                    {
                        "1897719318743327227": 2.0,
                    },
                    {
                        "1897719318743327227": 9.0,
                    },
                    {},
                ],
                [
                    {
                        "https://www.investopedia.com/terms/b/blockchain.asp": 5.0,
                    },
                    {
                        "https://www.investopedia.com/terms/b/blockchain.asp": 9.0,
                    },
                    {
                        "https://www.investopedia.com/terms/b/blockchain.asp": 5.0,
                    },
                    {},
                ],
            ],
        )
        self.assertEqual(penalties.tolist(), [0, 1, 1, 0])


if __name__ == "__main__":
    unittest.main()
