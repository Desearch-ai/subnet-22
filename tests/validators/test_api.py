from unittest.mock import patch, AsyncMock, Mock
import unittest
from fastapi.testclient import TestClient
import sys


sys.argv = [
    sys.argv[0],
    "--wallet.name",
    "validator",
    "--wandb.off",
    "--netuid",
    "41",
    "--wallet.hotkey",
    "default",
    "--subtensor.network",
    "test",
    "--neuron.run_random_miner_syn_qs_interval",
    "0",
    "--neuron.run_all_miner_syn_qs_interval",
    "0",
    "--neuron.offline",
    "--neuron.synthetic_disabled",
]


from neurons.validators.api import app
from desearch.protocol import Model, ResultType


sys.argv = [sys.argv[0]]


class TestAPI(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

        self.headers = {"access-key": "test"}

    @staticmethod
    async def mock_async_generator(items):
        for item in items:
            yield item

    @patch("neurons.validators.api.neu")
    def test_search(self, mock_neu):

        mock_organic = AsyncMock()
        mock_organic.return_value = self.mock_async_generator(["chunk1", "chunk2"])

        mock_neu.advanced_scraper_validator.organic = mock_organic

        payload = {
            "prompt": "What is blockchain?",
            "tools": ["Twitter Search"],
            "date_filter": "PAST_MONTH",
            "model": "HORIZON",
            "result_type": "LINKS_WITH_FINAL_SUMMARY",
        }
        response = self.client.post("/search", json=payload, headers=self.headers)

        self.assertEqual(response.status_code, 200)

        mock_organic.assert_called_once_with(
            {
                "content": payload["prompt"],
                "tools": payload["tools"],
                "date_filter": payload["date_filter"],
            },
            Model(payload["model"]),
            result_type=ResultType(payload["result_type"]),
        )


if __name__ == "__main__":
    unittest.main()
