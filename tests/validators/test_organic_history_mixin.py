import torch
import unittest
from unittest.mock import patch
from neurons.validators.organic_history_mixin import OrganicHistoryMixin


class MockUID:
    def __init__(self, uid):
        self.uid = uid

    def item(self):
        return self.uid

    def __eq__(self, value):
        return value == self.uid


class TestOrganicHistoryMixin(unittest.TestCase):
    def setUp(self):
        self.mixin = OrganicHistoryMixin()

    @patch("time.time", return_value=100000)
    def test_clean_organic_history(self, mock_time):
        self.mixin.organic_history = {
            1: [
                {
                    "start_time": mock_time(),
                    "response": "response1",
                    "task": "task1",
                    "event": {"name": "event1_name", "text": "event1_text"},
                }
            ],
            2: [
                {
                    "start_time": mock_time() - self.mixin.HISTORY_EXPIRY_TIME - 1,
                    "response": "response2",
                    "task": "task2",
                    "event": {"name": "event2_name", "text": "event2_text"},
                },
                {
                    "start_time": mock_time(),
                    "response": "response4",
                    "task": "task4",
                    "event": {"name": "event4_name", "text": "event4_text"},
                },
            ],
            3: [
                {
                    "start_time": 0,
                    "response": "response3",
                    "task": "task3",
                    "event": {"name": "event3_name", "text": "event3_text"},
                }
            ],
        }
        self.mixin._clean_organic_history()
        self.assertEqual(len(self.mixin.organic_history), 2)
        self.assertIn(1, self.mixin.organic_history)
        self.assertIn(2, self.mixin.organic_history)
        self.assertNotIn(3, self.mixin.organic_history)

        self.assertEqual(len(self.mixin.organic_history[1]), 1)
        self.assertEqual(len(self.mixin.organic_history[2]), 1)

    def test_save_organic_response(self):
        responses = ["response1", "response2"]
        uids = [torch.tensor([1]), torch.tensor([2])]
        tasks = ["task1", "task2"]
        event = {
            "name": ["event1_name", "event2_name"],
            "text": ["event1_text", "event2_text"],
        }
        self.mixin._save_organic_response(uids, responses, tasks, event, 1000)
        self.assertEqual(
            self.mixin.organic_history,
            {
                1: [
                    {
                        "start_time": 1000,
                        "response": "response1",
                        "task": "task1",
                        "event": {"name": "event1_name", "text": "event1_text"},
                    }
                ],
                2: [
                    {
                        "start_time": 1000,
                        "response": "response2",
                        "task": "task2",
                        "event": {"name": "event2_name", "text": "event2_text"},
                    }
                ],
            },
        )

        responses = ["response2_1", "response3"]
        uids = [torch.tensor([2]), torch.tensor([3])]
        tasks = ["task2_1", "task3"]
        event = {
            "name": ["event2_1_name", "event3_name"],
            "text": ["event2_1_text", "event3_text"],
        }
        self.mixin._save_organic_response(uids, responses, tasks, event, 2000)
        self.assertEqual(
            self.mixin.organic_history,
            {
                1: [
                    {
                        "start_time": 1000,
                        "response": "response1",
                        "task": "task1",
                        "event": {"name": "event1_name", "text": "event1_text"},
                    }
                ],
                2: [
                    {
                        "start_time": 1000,
                        "response": "response2",
                        "task": "task2",
                        "event": {"name": "event2_name", "text": "event2_text"},
                    },
                    {
                        "start_time": 2000,
                        "response": "response2_1",
                        "task": "task2_1",
                        "event": {"name": "event2_1_name", "text": "event2_1_text"},
                    },
                ],
                3: [
                    {
                        "start_time": 2000,
                        "response": "response3",
                        "task": "task3",
                        "event": {"name": "event3_name", "text": "event3_text"},
                    }
                ],
            },
        )

    @patch("time.time", return_value=100000)
    def test_get_latest_organic_responses(self, mock_time):
        organic_history = {
            1: [
                {
                    "start_time": mock_time(),
                    "response": "response1",
                    "task": "task1",
                    "event": {"name": "event1_name", "text": "event1_text"},
                }
            ],
            2: [
                {
                    "start_time": mock_time() - 1000,
                    "response": "response2",
                    "task": "task2",
                    "event": {"name": "event2_name", "text": "event2_text"},
                },
                {
                    "start_time": mock_time(),
                    "response": "response4",
                    "task": "task4",
                    "event": {"name": "event4_name", "text": "event4_text"},
                },
            ],
            3: [
                {
                    "start_time": mock_time(),
                    "response": "response3",
                    "task": "task3",
                    "event": {"name": "event3_name", "text": "event3_text"},
                }
            ],
        }
        self.mixin.organic_history = organic_history.copy()

        event, tasks, responses, uids = (
            self.mixin.get_latest_organic_responses().values()
        )

        self.assertEqual(
            event,
            {
                "name": ["event1_name", "event4_name", "event3_name"],
                "text": ["event1_text", "event4_text", "event3_text"],
            },
        )
        self.assertEqual(tasks, ["task1", "task4", "task3"])
        self.assertEqual(responses, ["response1", "response4", "response3"])
        self.assertEqual(
            uids.tolist(),
            [torch.tensor([1]), torch.tensor([2]), torch.tensor([3])],
        )

        self.assertEqual(self.mixin.organic_history, organic_history)

    @patch("time.time", return_value=100000)
    def test_get_uids_with_no_hitory(self, mock_time):
        self.mixin.organic_history = {
            1: [
                {
                    "start_time": mock_time(),
                    "response": "response1",
                    "task": "task1",
                    "event": {"name": "event1_name", "text": "event1_text"},
                }
            ],
            2: [
                {
                    "start_time": mock_time() - self.mixin.HISTORY_EXPIRY_TIME - 1,
                    "response": "response2",
                    "task": "task2",
                    "event": {"name": "event2_name", "text": "event2_text"},
                },
                {
                    "start_time": mock_time(),
                    "response": "response4",
                    "task": "task4",
                    "event": {"name": "event4_name", "text": "event4_text"},
                },
            ],
            3: [
                {
                    "start_time": 0,
                    "response": "response3",
                    "task": "task3",
                    "event": {"name": "event3_name", "text": "event3_text"},
                }
            ],
        }
        uids = self.mixin.get_uids_with_no_history([1, 2, 3, 4, 5])
        self.assertEqual(uids, [3, 4, 5])


if __name__ == "__main__":
    unittest.main()
