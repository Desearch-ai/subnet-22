import unittest

from desearch.protocol import ResultType, ScraperStreamingSynapse, ScraperTextRole
from neurons.validators.penalty.streaming_penalty import StreamingPenaltyModel
from tests.validators.penalty.text_chunks_data import TEXT_CHUNKS


class StreamingPenaltyModelTestCase(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.model = StreamingPenaltyModel(max_penalty=1.0, neuron=None)

    def _make_response(
        self,
        text_chunks,
        result_type: ResultType = ResultType.LINKS_WITH_FINAL_SUMMARY,
    ):
        return ScraperStreamingSynapse(
            prompt="What is blockchain?",
            text_chunks=text_chunks,
            result_type=result_type,
        )

    async def test_calculate_penalties_from_text_chunks(self):
        response = self._make_response(
            {
                ScraperTextRole.FINAL_SUMMARY: TEXT_CHUNKS,
            }
        )

        penalties = await self.model.calculate_penalties([response])

        self.assertAlmostEqual(
            penalties[0].item(),
            0,
        )

    async def test_calculate_penalties_returns_max_penalty_without_chunks(self):
        response = self._make_response({})

        penalties = await self.model.calculate_penalties([response])

        self.assertAlmostEqual(penalties[0].item(), 1.0)

    async def test_only_links_result_type_skips_penalty(self):
        response = self._make_response({}, result_type=ResultType.ONLY_LINKS)

        penalties = await self.model.calculate_penalties([response])

        self.assertAlmostEqual(penalties[0].item(), 0.0)


if __name__ == "__main__":
    unittest.main()
