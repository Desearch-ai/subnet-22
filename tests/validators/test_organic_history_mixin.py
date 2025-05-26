import time
import torch
import unittest
from unittest.mock import patch, AsyncMock

from neurons.validators.organic_history_mixin import OrganicHistoryMixin


class TestOrganicHistoryMixin(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        # Initialize mixin and stub out Redis methods
        self.mixin = OrganicHistoryMixin()
        self.mixin._save_history = AsyncMock()
        self.mixin._load_history = AsyncMock(return_value={})
        self.mixin._history_loaded = True

    @patch("time.time", return_value=100000)
    async def test_clean_organic_history(self, mock_time):
        # Prepare history with entries inside and outside expiry window
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
        # Clean expired entries
        cleaned = await self.mixin._clean_organic_history()

        # Only UIDs 1 and 2 remain
        self.assertEqual(set(self.mixin.organic_history.keys()), {1, 2})
        self.assertEqual(len(self.mixin.organic_history[1]), 1)
        self.assertEqual(len(self.mixin.organic_history[2]), 1)

    async def test_save_organic_response(self):
        # First batch
        responses = ["response1", "response2"]
        uids = [torch.tensor([1]), torch.tensor([2])]
        tasks = ["task1", "task2"]
        event = {
            "name": ["event1_name", "event2_name"],
            "text": ["event1_text", "event2_text"],
        }

        await self.mixin._save_organic_response(
            uids, responses, tasks, event, start_time=1000
        )
        expected = {
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
        }
        self.assertEqual(self.mixin.organic_history, expected)

        # Second batch
        responses = ["response2_1", "response3"]
        uids = [torch.tensor([2]), torch.tensor([3])]
        tasks = ["task2_1", "task3"]
        event = {
            "name": ["event2_1_name", "event3_name"],
            "text": ["event2_1_text", "event3_text"],
        }

        await self.mixin._save_organic_response(
            uids, responses, tasks, event, start_time=2000
        )
        expected[2].append(
            {
                "start_time": 2000,
                "response": "response2_1",
                "task": "task2_1",
                "event": {"name": "event2_1_name", "text": "event2_1_text"},
            }
        )
        expected[3] = [
            {
                "start_time": 2000,
                "response": "response3",
                "task": "task3",
                "event": {"name": "event3_name", "text": "event3_text"},
            }
        ]
        self.assertEqual(self.mixin.organic_history, expected)

    @patch("time.time", return_value=100000)
    async def test_get_random_organic_responses(self, mock_time):
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
        self.mixin._clean_organic_history = AsyncMock()

        # Fetch random
        result = await self.mixin.get_random_organic_responses()
        event = result["event"]
        tasks = result["tasks"]
        responses = result["responses"]
        uids = result["uids"]

        self.assertEqual(event["name"][0], "event1_name")
        self.assertIn(event["name"][1], ["event2_name", "event4_name"])
        self.assertEqual(event["name"][2], "event3_name")
        self.assertEqual(tasks[0], "task1")
        self.assertIn(tasks[1], ["task2", "task4"])
        self.assertEqual(tasks[2], "task3")
        self.assertEqual(responses[0], "response1")
        self.assertIn(responses[1], ["response2", "response4"])
        self.assertEqual(responses[2], "response3")
        # UIDs should remain consistent [1,2,3]
        self.assertEqual(uids.tolist(), [1, 2, 3])

    @patch("time.time", return_value=100000)
    async def test_get_uids_with_no_history(self, mock_time):
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
        self.mixin._clean_organic_history = AsyncMock()

        uids = await self.mixin.get_uids_with_no_history([1, 2, 3, 4, 5])
        self.assertEqual(uids, [4, 5])


if __name__ == "__main__":
    unittest.main()
