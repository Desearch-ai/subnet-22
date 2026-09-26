import unittest
from unittest.mock import AsyncMock, Mock, patch

import numpy as np

from desearch.bittensor.metagraph import generateMockNeurons
from desearch.bittensor.wallet import MOCK_WALLET_KEY
from neurons.validators.scoring.weights import burn_weights, set_weights

WEIGHTS = "neurons.validators.scoring.weights"


class TestWeights(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.neuron = Mock()
        self.neuron.config.netuid = 22
        self.neuron.metagraph.neurons = generateMockNeurons(4)
        self.neuron.metagraph.uids = np.array([0, 1, 2, 3], dtype=np.int64)
        self.neuron.metagraph.n = 4
        self.neuron.moving_averaged_scores = np.zeros(4, dtype=np.float32)
        self.neuron.subtensor.min_allowed_weights = AsyncMock(return_value=1)
        self.neuron.subtensor.max_weight_limit = AsyncMock(return_value=1.0)
        self.neuron.subtensor.set_weights = AsyncMock(return_value=(True, "ok"))

    @patch(f"{WEIGHTS}.EMISSION_CONTROL_HOTKEY", "hotkey2")
    def test_burn_weights_puts_everything_on_burn_uid(self):
        np.testing.assert_array_equal(burn_weights(self.neuron), [0, 0, 1, 0])

    @patch(f"{WEIGHTS}.EMISSION_CONTROL_HOTKEY", MOCK_WALLET_KEY)
    def test_burn_weights_on_uid_zero(self):
        np.testing.assert_array_equal(burn_weights(self.neuron), [1, 0, 0, 0])

    @patch(f"{WEIGHTS}.EMISSION_CONTROL_HOTKEY", "missing")
    def test_burn_weights_without_burn_hotkey(self):
        self.assertIsNone(burn_weights(self.neuron))

    @patch(f"{WEIGHTS}.EMISSION_CONTROL_HOTKEY", "hotkey2")
    async def test_set_weights_sends_only_burn_uid_with_zero_scores(self):
        self.assertTrue(await set_weights(self.neuron))

        kwargs = self.neuron.subtensor.set_weights.await_args.kwargs
        np.testing.assert_array_equal(kwargs["uids"], [2])
        np.testing.assert_array_equal(kwargs["weights"], [1.0])

    @patch(f"{WEIGHTS}.EMISSION_CONTROL_HOTKEY", "missing")
    async def test_set_weights_skips_without_burn_hotkey(self):
        self.assertFalse(await set_weights(self.neuron))
        self.neuron.subtensor.set_weights.assert_not_awaited()


if __name__ == "__main__":
    unittest.main()
