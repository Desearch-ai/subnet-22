import unittest

from bittensor import TerminalInfo

from desearch.protocol import TwitterSearchSynapse
from neurons.validators.penalty.min_realistic_time_penalty import (
    MinRealisticTimePenaltyModel,
)


def _response(process_time):
    """Build a minimal synapse with a dendrite that exposes process_time."""
    r = TwitterSearchSynapse(query="x", results=[])
    r.dendrite = TerminalInfo(process_time=process_time)
    return r


class MinRealisticTimePenaltyTestCase(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.model = MinRealisticTimePenaltyModel(min_realistic_time=1.0)

    async def test_below_min_full_penalty(self):
        penalties = await self.model.calculate_penalties([_response(0.5)])
        self.assertEqual(penalties.tolist(), [1.0])

    async def test_exactly_at_min_passes(self):
        penalties = await self.model.calculate_penalties([_response(1.0)])
        self.assertEqual(penalties.tolist(), [0])

    async def test_above_min_passes(self):
        penalties = await self.model.calculate_penalties([_response(2.5)])
        self.assertEqual(penalties.tolist(), [0])

    async def test_missing_process_time_no_penalty(self):
        """TimeoutPenalty handles None; this one stays out of its lane."""
        penalties = await self.model.calculate_penalties([_response(None)])
        self.assertEqual(penalties.tolist(), [0])

    async def test_string_numeric_parses(self):
        """0.5 as string is still sub-realistic."""
        penalties = await self.model.calculate_penalties([_response("0.5")])
        self.assertEqual(penalties.tolist(), [1.0])

    async def test_different_thresholds(self):
        ai = MinRealisticTimePenaltyModel(min_realistic_time=5.0)
        web = MinRealisticTimePenaltyModel(min_realistic_time=0.7)
        # 2.0s passes for X (min=1.0) but fails for AI (min=5.0)
        self.assertEqual(
            (await ai.calculate_penalties([_response(2.0)])).tolist(), [1.0]
        )
        self.assertEqual(
            (await web.calculate_penalties([_response(2.0)])).tolist(), [0]
        )

    async def test_batch(self):
        responses = [_response(0.3), _response(2.0), _response(None), _response(0.8)]
        penalties = await self.model.calculate_penalties(responses)
        self.assertEqual(penalties.tolist(), [1.0, 0, 0, 1.0])


if __name__ == "__main__":
    unittest.main()
