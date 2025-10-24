import torch
from typing import List
from neurons.validators.base_validator import AbstractNeuron
from neurons.validators.utils.tasks import Task
from neurons.validators.penalty.penalty import BasePenaltyModel, PenaltyModelType
import bittensor as bt
from desearch.protocol import TwitterSearchSynapse

MAX_PENALTY = 1.0


class TwitterCountPenaltyModel(BasePenaltyModel):
    def __init__(self, max_penalty: float = MAX_PENALTY, neuron: AbstractNeuron = None):
        super().__init__(max_penalty, neuron)

    @property
    def name(self) -> str:
        return PenaltyModelType.twitter_count_penalty.value

    async def calculate_penalties(
        self,
        responses: List[TwitterSearchSynapse],
        tasks: List[Task],
        additional_params=None,
    ) -> torch.FloatTensor:

        penalties = torch.zeros(len(responses), dtype=torch.float32)

        for index, response in enumerate(responses):
            if not isinstance(response, TwitterSearchSynapse):
                penalties[index] = 0.0
                bt.logging.debug(
                    f"Response index {index} is not TwitterSearchSynapse. No penalty."
                )
                continue

            results_count = len(response.results)

            if results_count > response.count:
                penalties[index] = 0
            else:
                penalties[index] = 1 - results_count / response.count

            bt.logging.debug(f"Response index {index} has penalty {penalties[index]}")

        return penalties
