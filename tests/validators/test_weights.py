from unittest.mock import Mock, patch
import unittest
import numpy as np
from neurons.validators.scoring.weights import burn_weights
from desearch.bittensor.metagraph import generateMockNeurons


class TestWeights(unittest.TestCase):
    def setUp(self):
        self.neuron = Mock()
        self.neuron.metagraph.neurons = generateMockNeurons(4)
        self.neuron.metagraph.uids = np.array([0, 1, 2, 3], dtype=np.int64)

    @patch("neurons.validators.scoring.weights.EMISSION_CONTROL_HOTKEY", "hotkey1")
    def test_burn_weights(self):
        weights = burn_weights(
            self.neuron, np.array([0, 1, 1, 1], dtype=np.float32)
        )
        self.assertAlmostEqual(weights[0], 0.0, places=5)
        self.assertAlmostEqual(weights[1], 2.4, places=5)
        self.assertAlmostEqual(weights[2], 0.3, places=5)
        self.assertAlmostEqual(weights[3], 0.3, places=5)


if __name__ == "__main__":
    unittest.main()
