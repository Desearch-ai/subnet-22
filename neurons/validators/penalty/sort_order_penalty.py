from typing import List

import bittensor as bt
import numpy as np

from desearch.protocol import TwitterSearchSynapse
from neurons.validators.base_validator import AbstractNeuron
from neurons.validators.penalty.penalty import BasePenaltyModel, PenaltyModelType
from neurons.validators.utils.response_checks import is_descending_by_created_at

MAX_PENALTY = 1.0


class SortOrderPenaltyModel(BasePenaltyModel):
    """Penalize Twitter responses with sort=Latest that aren't sorted by
    created_at descending."""

    is_deep = False

    def __init__(self, max_penalty: float = MAX_PENALTY, neuron: AbstractNeuron = None):
        super().__init__(max_penalty, neuron)

    @property
    def name(self) -> str:
        return PenaltyModelType.sort_order_penalty.value

    async def calculate_penalties(
        self,
        responses: List[bt.Synapse],
        additional_params=None,
    ) -> np.ndarray:
        penalties = np.zeros(len(responses), dtype=np.float32)
        for i, response in enumerate(responses):
            if not isinstance(response, TwitterSearchSynapse):
                continue
            if getattr(response, "sort", None) != "Latest":
                continue
            if not is_descending_by_created_at(response.results or []):
                penalties[i] = self.max_penalty
        return penalties
