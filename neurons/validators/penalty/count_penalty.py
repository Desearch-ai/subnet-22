from typing import List, Optional

import bittensor as bt
import torch

from desearch.protocol import TwitterSearchSynapse, WebSearchSynapse
from neurons.validators.base_validator import AbstractNeuron
from neurons.validators.penalty.penalty import BasePenaltyModel, PenaltyModelType

MAX_PENALTY = 1.0


class CountPenaltyModel(BasePenaltyModel):
    """Penalize miners that return fewer results than the validator requested.

    Twitter uses ``count`` and Web uses ``num`` for the requested-results field;
    both expose ``results`` for the actual list. Other synapse types are skipped.
    """

    def __init__(self, max_penalty: float = MAX_PENALTY, neuron: AbstractNeuron = None):
        super().__init__(max_penalty, neuron)

    @property
    def name(self) -> str:
        return PenaltyModelType.count_penalty.value

    @staticmethod
    def _requested_count(response) -> Optional[int]:
        if isinstance(response, TwitterSearchSynapse):
            return response.count
        if isinstance(response, WebSearchSynapse):
            return response.num
        return None

    async def calculate_penalties(
        self,
        responses: List[bt.Synapse],
        additional_params=None,
    ) -> torch.FloatTensor:
        penalties = torch.zeros(len(responses), dtype=torch.float32)

        for index, response in enumerate(responses):
            requested = self._requested_count(response)
            if requested is None or requested <= 0:
                penalties[index] = 0.0
                bt.logging.debug(
                    f"Response index {index} has no countable request field. No penalty."
                )
                continue

            results_count = len(response.results or [])

            if results_count >= requested:
                penalties[index] = 0
            else:
                penalties[index] = 1 - results_count / requested

            bt.logging.debug(f"Response index {index} has penalty {penalties[index]}")

        return penalties
