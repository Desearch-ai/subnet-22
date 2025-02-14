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
            1: {
                "start_time": mock_time(),
                "response": "response1",
                "task": "task1",
                "event": {"name": "event1_name", "text": "event1_text"},
            },
            2: {
                "start_time": mock_time() - self.mixin.HISTORY_EXPIRY_TIME - 1,
                "response": "response2",
                "task": "task2",
                "event": {"name": "event2_name", "text": "event2_text"},
            },
            3: {
                "start_time": 0,
                "response": "response3",
                "task": "task3",
                "event": {"name": "event3_name", "text": "event3_text"},
            },
        }

        self.mixin._clean_organic_history()

        self.assertEqual(len(self.mixin.organic_history), 1)
        self.assertIn(1, self.mixin.organic_history)
        self.assertNotIn(2, self.mixin.organic_history)
        self.assertNotIn(3, self.mixin.organic_history)

    @patch("time.time", return_value=100000)
    def test_merge_synthetic_organic_responses(self, mock_time):

        responses = ["response2", "response5"]
        uids = [MockUID(2), MockUID(5)]
        tasks = ["task2", "task5"]
        event = {
            "name": ["event2_name", "event5_name"],
            "text": ["event2_text", "event5_text"],
        }
        available_uids = [MockUID(i) for i in range(1, 6)]

        self.mixin.organic_history = {
            1: {
                "start_time": mock_time() - 1000,
                "response": "response1",
                "task": "task1",
                "event": {"name": "event1_name", "text": "event1_text"},
            },
            3: {
                "start_time": mock_time() - 2000,
                "response": "response3",
                "task": "task3",
                "event": {"name": "event3_name", "text": "event3_text"},
            },
            4: {
                "start_time": mock_time() - 10000,
                "response": "response4",
                "task": "task4",
                "event": {"name": "event4_name", "text": "event4_text"},
            },
        }

        merged_event, merged_tasks, merged_responses, merged_uids, start_time = (
            self.mixin._merge_synthetic_organic_responses(
                responses, uids, tasks, event, mock_time(), available_uids
            )
        )

        self.assertEqual(
            merged_event,
            {
                "name": [f"event{i}_name" for i in range(1, 6)],
                "text": [f"event{i}_text" for i in range(1, 6)],
            },
        )
        self.assertEqual(merged_tasks, [f"task{i}" for i in range(1, 6)])
        self.assertEqual(merged_responses, [f"response{i}" for i in range(1, 6)])
        self.assertEqual(merged_uids, [MockUID(i) for i in range(1, 6)])
        self.assertEqual(start_time, mock_time())

    def test_save_organic_response(self):
        responses = ["response1", "response2"]
        uids = [MockUID(1), MockUID(2)]
        tasks = ["task1", "task2"]
        event = {
            "name": ["event1_name", "event2_name"],
            "text": ["event1_text", "event2_text"],
        }
        self.mixin._save_organic_response(uids, responses, tasks, event, 1000)

        self.assertEqual(
            self.mixin.organic_history,
            {
                1: {
                    "start_time": 1000,
                    "response": "response1",
                    "task": "task1",
                    "event": {"name": "event1_name", "text": "event1_text"},
                },
                2: {
                    "start_time": 1000,
                    "response": "response2",
                    "task": "task2",
                    "event": {"name": "event2_name", "text": "event2_text"},
                },
            },
        )


if __name__ == "__main__":
    unittest.main()
