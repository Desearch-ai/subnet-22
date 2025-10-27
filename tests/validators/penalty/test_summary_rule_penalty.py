import unittest
from neurons.validators.penalty.summary_rule_penalty import SummaryRulePenaltyModel
from desearch.protocol import ScraperStreamingSynapse, ScraperTextRole
import torch


class SummaryRulePenaltyTestCase(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.model = SummaryRulePenaltyModel()

    async def test_calculate_penalties_with_no_system_message(self):
        penalties = await self.model.calculate_penalties(
            [ScraperStreamingSynapse(prompt="What is blockchain?")], []
        )
        self.assertEqual(penalties, torch.tensor([0]))

    async def test_calculate_penalties_with_system_message_no_penalty(self):
        penalties = await self.model.calculate_penalties(
            [
                ScraperStreamingSynapse(
                    prompt="What is blockchain?",
                    system_message="Summarize the content by categorizing key points into 'Pros' and 'Cons' sections.",
                    text_chunks={
                        ScraperTextRole.FINAL_SUMMARY: [
                            """ **Summary**
                        **Pros:**
                          Blockchain technology provides decentralization, transparency, and security, offering significant benefits across industries like finance, supply chain management, and healthcare. It facilitates smart contract implementation, ensures data integrity, and showcases practical applications in areas such as logistics and voting systems.
                        **Cons:**
                          One of the challenges faced by blockchain networks is scalability issues. However, continuous advancements are being made to improve network efficiency and tackle interoperability concerns, aiming for a more interconnected blockchain ecosystem."""
                        ]
                    },
                )
            ],
            [],
        )
        self.assertEqual(penalties, torch.tensor([0]))

    async def test_calculate_penalties_with_system_message_penalty(self):
        penalties = await self.model.calculate_penalties(
            [
                ScraperStreamingSynapse(
                    prompt="What is blockchain?",
                    system_message="Summarize the content by categorizing key points into 'Pros' and 'Cons' sections.",
                    text_chunks={
                        ScraperTextRole.FINAL_SUMMARY: [
                            """ **Summary**
                          Blockchain technology provides decentralization, transparency, and security, offering significant benefits across industries like finance, supply chain management, and healthcare. It facilitates smart contract implementation, ensures data integrity, and showcases practical applications in areas such as logistics and voting systems.
                          One of the challenges faced by blockchain networks is scalability issues. However, continuous advancements are being made to improve network efficiency and tackle interoperability concerns, aiming for a more interconnected blockchain ecosystem."""
                        ]
                    },
                )
            ],
            [],
        )
        self.assertEqual(penalties, torch.tensor([0.2]))


if __name__ == "__main__":
    unittest.main()
