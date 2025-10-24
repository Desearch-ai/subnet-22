import unittest
import torch
import math
from typing import List
from neurons.validators.utils.tasks import TwitterTask
from desearch.protocol import ScraperStreamingSynapse  # Ensure this import is correct
from neurons.validators.penalty.exponential_penalty import ExponentialTimePenaltyModel


class MockDendrite:
    """
    A mock class to simulate the dendrite attribute with dictionary-like access.
    """

    def __init__(self, process_time):
        self.process_time = process_time

    def get(self, key, default=None):
        if key == "process_time":
            return self.process_time
        return default


class MockResponse:
    """
    A mock class to simulate ScraperStreamingSynapse responses.
    """

    def __init__(self, process_time, max_execution_time=10):
        self.dendrite = MockDendrite(process_time)
        self.max_execution_time = max_execution_time


class ExponentialTimePenaltyModelTestCase(unittest.TestCase):
    def setUp(self):
        """
        Initialize the ExponentialTimePenaltyModel with a fixed max_execution_time and max_penalty.
        """
        self.max_execution_time = 10  # in seconds
        self.max_penalty = 1.0
        self.penalty_model = ExponentialTimePenaltyModel(max_penalty=self.max_penalty)
        print(
            f"Initialized ExponentialTimePenaltyModel with max_execution_time={self.max_execution_time} and max_penalty={self.max_penalty}"
        )

    def test_no_penalty_if_within_time(self):
        """
        Test that no penalty is applied when process_time is within or exactly at the max_execution_time.
        """
        print("\nRunning test_no_penalty_if_within_time")
        responses = [
            self.create_mock_response(process_time=5),  # well within limit
            self.create_mock_response(process_time=10),  # exactly at the limit
        ]
        tasks = [
            TwitterTask(
                base_text="Test", task_name="test", task_type="test", criteria=[]
            )
        ] * len(responses)

        penalties = self.penalty_model.calculate_penalties(responses, tasks)
        print(f"Calculated penalties: {penalties.tolist()}")

        for idx, p in enumerate(penalties):
            process_time = responses[idx].dendrite.get("process_time")
            penalty = p.item()
            print(
                f"max_execution_time={self.max_execution_time}, process_time={process_time}, penalty={penalty}"
            )
            self.assertEqual(
                penalty,
                0.0,
                "No penalty expected if process_time <= max_execution_time.",
            )

    def test_penalty_if_exceeds_time(self):
        """
        Test that a penalty is applied when process_time exceeds the max_execution_time.
        """
        print("\nRunning test_penalty_if_exceeds_time")
        responses = [
            self.create_mock_response(process_time=20),
        ]
        tasks = [
            TwitterTask(
                base_text="Test", task_name="test", task_type="test", criteria=[]
            )
        ]

        penalties = self.penalty_model.calculate_penalties(responses, tasks)
        print(f"Calculated penalties: {penalties.tolist()}")

        process_time = responses[0].dendrite.get("process_time")
        penalty = penalties[0].item()
        print(
            f"max_execution_time={self.max_execution_time}, process_time={process_time}, penalty={penalty}"
        )
        # With a large delay (20-10=10s), penalty ~ 1.0
        self.assertAlmostEqual(
            penalty, 1.0, delta=0.001, msg="Penalty should be near max for large delay."
        )

    def test_partial_penalty_for_slight_exceed(self):
        """
        Test that a partial penalty is applied when process_time slightly exceeds the max_execution_time.
        """
        print("\nRunning test_partial_penalty_for_slight_exceed")
        responses = [
            self.create_mock_response(process_time=11),
        ]
        tasks = [
            TwitterTask(
                base_text="Test", task_name="test", task_type="test", criteria=[]
            )
        ]

        penalties = self.penalty_model.calculate_penalties(responses, tasks)
        expected_penalty = 1 - math.exp(-1)  # about 0.63212
        print(f"Calculated penalties: {penalties.tolist()}")

        process_time = responses[0].dendrite.get("process_time")
        penalty = penalties[0].item()
        print(
            f"max_execution_time={self.max_execution_time}, process_time={process_time}, penalty={penalty}"
        )

        self.assertAlmostEqual(
            penalty,
            expected_penalty,
            delta=0.001,
            msg="Penalty should match exponential formula.",
        )

    def create_mock_response(self, process_time):
        # Create a MockResponse with process_time
        response = MockResponse(process_time)
        print(f"Created mock response with process_time={process_time}")
        return response


if __name__ == "__main__":
    unittest.main()
