from unittest.mock import Mock, patch
import unittest
from neurons.validators.weights import burn_weights
from datura.bittensor.metagraph import generateMockNeurons
import torch


class TestWeights(unittest.TestCase):
    def setUp(self):
        self.neuron = Mock()
        self.neuron.metagraph.neurons = generateMockNeurons(4)
        self.neuron.metagraph.uids = torch.tensor([0, 1, 2, 3])

    @patch("neurons.validators.weights.EMISSION_CONTROL_HOTKEY", "hotkey1")
    def test_burn_weights(self):
        weights = burn_weights(self.neuron, torch.tensor([0, 1, 1, 1]))
        self.assertEqual(weights[0], torch.tensor([0]))
        self.assertEqual(weights[1], torch.tensor([2.4]))
        self.assertEqual(weights[2], torch.tensor([0.3]))
        self.assertEqual(weights[3], torch.tensor([0.3]))


if __name__ == "__main__":
    unittest.main()
