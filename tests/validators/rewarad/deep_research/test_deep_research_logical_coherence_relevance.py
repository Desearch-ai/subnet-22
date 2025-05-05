import unittest
from neurons.validators.reward.deep_research_logical_coherence import (
    DeepResearchLogicalCoherenceRelevanceModel,
)
from datura.protocol import DeepResearchSynapse
from tests_data.reports.what_is_blockchain import report_what_is_blockchain


class DeepResearchLogicalCoherenceRelevanceModelTestCase(
    unittest.IsolatedAsyncioTestCase
):
    def setUp(self):
        self.device = "test_device"
        self.scoring_type = None
        self.model = DeepResearchLogicalCoherenceRelevanceModel(
            self.device, self.scoring_type
        )

    async def test_check_response(self):
        report = report_what_is_blockchain
        score = await self.model.check_response(
            DeepResearchSynapse(prompt="what is blockchain", report=report)
        )
        self.assertEqual(score, 1)

        report.append(
            {
                "title": "Components of blockchain",
                "description": """The key components of blockchain are:
Blocks: Units of data that store transaction information, timestamps, and unique hashes.
Chain: The sequential connection of blocks, ensuring immutability through hashes.
Decentralized Network: A distributed network of nodes that maintains the blockchain, ensuring no central authority.
Distributed Ledger: A shared and synchronized ledger across all network participants.
Consensus Mechanism: Protocols like Proof of Work or Proof of Stake that validate transactions across the network.
Cryptography: Techniques like hashing and digital signatures to secure data and transactions.
Smart Contracts: Self-executing contracts that automate processes when predefined conditions are met.""",
            }
        )
        score = await self.model.check_response(
            DeepResearchSynapse(prompt="what is blockchain", report=report)
        )
        self.assertEqual(score, 0.2)

        report = report_what_is_blockchain
        report[3] = {
            "title": "Consensus Mechanisms and Their Impact",
            "description": "",
        }
        score = await self.model.check_response(
            DeepResearchSynapse(prompt="what is blockchain", report=report)
        )
        self.assertEqual(score, 0.5)


if __name__ == "__main__":
    unittest.main()
