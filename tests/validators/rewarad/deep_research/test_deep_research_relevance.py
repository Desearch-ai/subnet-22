import unittest
from neurons.validators.reward.deep_research_relevance import (
    DeepResearchContentRelevanceModel,
)
from tests_data.reports.what_is_blockchain import report_what_is_blockchain


class DeepResearchRelevanceModelTestCase(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.device = "test_device"
        self.scoring_type = None
        self.model = DeepResearchContentRelevanceModel(self.device, self.scoring_type)

    async def test_check_section_relevance(self):
        report = report_what_is_blockchain
        score = await self.model.check_section_relevance(
            prompt="what is blockchain?", section=report[0]
        )
        self.assertEqual(score, 1)

        score = await self.model.check_section_relevance(
            prompt="ai startup companies in us?", section=report[1]
        )
        self.assertEqual(score, 0.2)

        score = await self.model.check_section_relevance(
            prompt="how to learn blockchain?", section=report[2]
        )
        self.assertEqual(score, 0.5)


if __name__ == "__main__":
    unittest.main()
