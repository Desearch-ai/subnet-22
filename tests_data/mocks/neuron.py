from unittest.mock import Mock, patch
from datura.bittensor.metagraph import generateMockNeurons
import torch


def mock_neuron():
    neuron = Mock()
    neuron.metagraph.neurons = generateMockNeurons(4)
    neuron.metagraph.uids = torch.tensor([0, 1, 2, 3])
    neuron.config.neuron.device = "cpu"
    neuron.config.reward.performance_weight = 0.05

    return neuron
