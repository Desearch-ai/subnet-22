import torch
from typing import List
from neurons.validators.base_validator import AbstractNeuron
from neurons.validators.utils.tasks import Task
from neurons.validators.penalty.penalty import BasePenaltyModel, PenaltyModelType
import bittensor as bt
import math

MAX_PENALTY = 1.0


class ExponentialTimePenaltyModel(BasePenaltyModel):
    def __init__(self, max_penalty: float = MAX_PENALTY, neuron: AbstractNeuron = None):
        super().__init__(max_penalty, neuron)
        bt.logging.debug(
            "Initialized ExponentialTimePenaltyModel using max_execution_time from responses."
        )

    @property
    def name(self) -> str:
        return PenaltyModelType.exponential_penalty.value

    async def calculate_penalties(
        self,
        responses: List[bt.Synapse],
        tasks: List[Task],
        additional_params=None,
    ) -> torch.FloatTensor:

        penalties = torch.zeros(len(responses), dtype=torch.float32)

        for index, response in enumerate(responses):
            process_time = getattr(response.dendrite, "process_time", None)
            max_execution_time = getattr(response, "max_execution_time", None)

            # Log the retrieved values for debugging
            # bt.logging.info(f"Full Response: {response}")

            # If you want to inspect the full dendrite object
            # bt.logging.info(f"Dendrite: {response.dendrite}")

            if not process_time:
                penalties[index] = self.max_penalty
                bt.logging.debug(
                    f"Response index {index} has no process time. Penalty: {self.max_penalty}"
                )
            elif process_time <= max_execution_time:
                # No penalty if processed within the allowed time
                penalties[index] = 0.0
                bt.logging.debug(
                    f"Response index {index} processed in {process_time}s (<= {max_execution_time}s). No penalty."
                )
            else:
                # Calculate penalty for exceeding allowed time
                delay = process_time - max_execution_time
                penalty = 1 - math.exp(-delay)
                penalty = min(penalty, self.max_penalty)
                penalties[index] = penalty
                bt.logging.debug(
                    f"Response index {index} exceeded allowed time by {delay}s. Penalty: {penalty}"
                )

        return penalties
