import unittest
from unittest.mock import Mock, patch

import numpy as np

from desearch.bittensor.metagraph import generateMockNeurons
from neurons.validators.scoring.weights import burn_weights


class TestWeights(unittest.TestCase):
    def setUp(self):
        self.neuron = Mock()
        self.neuron.metagraph.neurons = generateMockNeurons(4)
        self.neuron.metagraph.uids = np.array([0, 1, 2, 3], dtype=np.int64)

    @patch("neurons.validators.scoring.weights.EMISSION_CONTROL_HOTKEY", "hotkey1")
    def test_burn_weights(self):
        weights = burn_weights(self.neuron, np.array([0, 1, 1, 1], dtype=np.float32))
        self.assertAlmostEqual(weights[0], 0.0, places=5)
        self.assertAlmostEqual(weights[1], 2.1, places=5)
        self.assertAlmostEqual(weights[2], 0.45, places=5)
        self.assertAlmostEqual(weights[3], 0.45, places=5)


if __name__ == "__main__":
    unittest.main()
