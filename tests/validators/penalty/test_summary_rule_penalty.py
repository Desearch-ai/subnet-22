import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from neurons.validators.penalty.summary_rule_penalty import SummaryRulePenaltyModel
from desearch.protocol import ScoringModel, ScraperStreamingSynapse, ScraperTextRole


def _neuron(scoring_model=ScoringModel.QWEN3_32B):
    return SimpleNamespace(
        config=SimpleNamespace(neuron=SimpleNamespace(scoring_model=scoring_model))
    )


class SummaryRulePenaltyTestCase(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.model = SummaryRulePenaltyModel(neuron=_neuron())

    async def test_calculate_penalties_with_no_system_message(self):
        penalties = await self.model.calculate_penalties(
            [ScraperStreamingSynapse(prompt="What is blockchain?")], []
        )
        self.assertEqual(penalties.tolist(), [0])

    @patch(
        "neurons.validators.penalty.summary_rule_penalty.call_scoring_llm",
        new_callable=AsyncMock,
    )
    async def test_calculate_penalties_with_system_message_no_penalty(self, mock_llm):
        mock_llm.return_value = "Score 10: fully adheres to the rule"
        penalties = await self.model.calculate_penalties(
            [
                ScraperStreamingSynapse(
                    prompt="What is blockchain?",
                    system_message="Summarize the content by categorizing key points into 'Pros' and 'Cons' sections.",
                    text_chunks={
                        ScraperTextRole.FINAL_SUMMARY: ["**Pros:** ...\n**Cons:** ..."]
                    },
                )
            ],
            [],
        )
        self.assertEqual(penalties.tolist(), [0])
        mock_llm.assert_awaited_once()
        self.assertEqual(mock_llm.await_args.kwargs["model"], ScoringModel.QWEN3_32B)

    @patch(
        "neurons.validators.penalty.summary_rule_penalty.call_scoring_llm",
        new_callable=AsyncMock,
    )
    async def test_calculate_penalties_with_system_message_penalty(self, mock_llm):
        mock_llm.return_value = "Score 0: does not follow the required structure"
        penalties = await self.model.calculate_penalties(
            [
                ScraperStreamingSynapse(
                    prompt="What is blockchain?",
                    system_message="Summarize the content by categorizing key points into 'Pros' and 'Cons' sections.",
                    text_chunks={
                        ScraperTextRole.FINAL_SUMMARY: ["A flat summary with no sections."]
                    },
                )
            ],
            [],
        )
        self.assertAlmostEqual(penalties.tolist()[0], 1.0, places=5)


if __name__ == "__main__":
    unittest.main()
