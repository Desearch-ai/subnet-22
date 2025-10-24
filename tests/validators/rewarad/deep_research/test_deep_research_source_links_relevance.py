import unittest
from neurons.validators.reward.deep_research_source_links import (
    DeepResearchSourceLinksRelevanceModel,
)
from desearch.protocol import ReportItem
from tests_data.reports.what_is_blockchain import report_what_is_blockchain


class DeepResearchSourceLinksRelevanceModelTestCase(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.device = "test_device"
        self.scoring_type = None
        self.model = DeepResearchSourceLinksRelevanceModel(
            self.device, self.scoring_type
        )

    async def test_check_section_links(self):
        report = report_what_is_blockchain
        score = await self.model.check_section_links(
            prompt="what is blockchain?",
            section=ReportItem(**report[1]["subsections"][0]),
        )
        self.assertEqual(score, 1)

        score = await self.model.check_section_links(
            prompt="ai startup companies in us?",
            section=ReportItem(**report[1]["subsections"][1]),
        )
        self.assertEqual(score, 0.2)


if __name__ == "__main__":
    unittest.main()
