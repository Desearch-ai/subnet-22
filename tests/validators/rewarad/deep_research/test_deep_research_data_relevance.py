import unittest
from neurons.validators.reward.deep_research_data import (
    DeepResearchDataRelevanceModel,
)
from tests_data.reports.what_is_blockchain import report_what_is_blockchain
from datura.protocol import ReportItem


class DeepResearchDataRelevanceModelTestCase(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.device = "test_device"
        self.scoring_type = None
        self.model = DeepResearchDataRelevanceModel(self.device, self.scoring_type)

    async def test_check_section_data(self):
        report = report_what_is_blockchain
        score = await self.model.check_section_data(
            ReportItem(**report[1]["subsections"][0])
        )
        self.assertEqual(score, 1)

        score = await self.model.check_section_data(
            ReportItem(
                title="Components of blockchain",
                description="The blockchain network consists of components like nodes, consensus mechanisms, and smart contracts. The nodes work together to validate transactions, while the decentralized ledger records the activities. The consensus mechanism ensures all transactions are valid, and smart contracts are used to automate processes. The network's security is guaranteed by cryptographic techniques, making it secure for transactions. The blockchain was created in 1999 as a secure alternative to traditional databases.",
                links=[
                    "https://www.geeksforgeeks.org/components-of-blockchain-network/"
                ],
            )
        )
        self.assertEqual(score, 0.2)



if __name__ == "__main__":
    unittest.main()
