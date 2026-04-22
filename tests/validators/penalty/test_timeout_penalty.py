import unittest

from neurons.validators.penalty.timeout_penalty import TimeoutPenaltyModel


class MockDendrite:
    def __init__(self, process_time):
        self.process_time = process_time


class MockResponse:
    def __init__(self, process_time, max_execution_time=10, timeout=15):
        self.dendrite = MockDendrite(process_time)
        self.max_execution_time = max_execution_time
        self.timeout = timeout


class TimeoutPenaltyModelTestCase(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.penalty_model = TimeoutPenaltyModel(max_penalty=1.0)

    async def test_no_penalty_if_within_time(self):
        responses = [
            MockResponse(process_time=10, max_execution_time=10, timeout=15),
            MockResponse(process_time=9.5, max_execution_time=10, timeout=15),
        ]

        penalties = await self.penalty_model.calculate_penalties(responses)

        self.assertEqual(penalties.tolist(), [0.0, 0.0])

    async def test_penalty_grows_by_second_buckets(self):
        responses = [
            MockResponse(process_time=16, max_execution_time=15, timeout=20),
            MockResponse(process_time=17, max_execution_time=15, timeout=20),
        ]

        penalties = await self.penalty_model.calculate_penalties(responses)

        self.assertAlmostEqual(penalties[0].item(), 0.2)
        self.assertAlmostEqual(penalties[1].item(), 0.4)

    async def test_penalty_does_not_use_fractional_growth_inside_bucket(self):
        responses = [
            MockResponse(process_time=12.1, max_execution_time=10, timeout=15),
        ]

        penalties = await self.penalty_model.calculate_penalties(responses)

        self.assertAlmostEqual(penalties[0].item(), 0.6)

    async def test_penalty_caps_at_max_penalty(self):
        responses = [
            MockResponse(process_time=21, max_execution_time=15, timeout=20),
        ]

        penalties = await self.penalty_model.calculate_penalties(responses)

        self.assertAlmostEqual(penalties[0].item(), 1.0)

    async def test_uses_response_timeout_window_dynamically(self):
        responses = [
            MockResponse(process_time=12.3, max_execution_time=10, timeout=20),
        ]

        penalties = await self.penalty_model.calculate_penalties(responses)

        self.assertAlmostEqual(penalties[0].item(), 0.3)

    async def test_falls_back_to_configured_grace_window(self):
        penalty_model = TimeoutPenaltyModel(max_penalty=1.0, timeout_grace_seconds=5)
        response = MockResponse(process_time=11, max_execution_time=10, timeout=None)

        penalties = await penalty_model.calculate_penalties([response])

        self.assertAlmostEqual(penalties[0].item(), 0.2)
