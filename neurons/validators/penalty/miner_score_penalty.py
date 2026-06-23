from typing import List

import bittensor as bt
import numpy as np

from desearch.protocol import (
    ContextualRelevance,
    ScraperStreamingSynapse,
)
from neurons.validators.base_validator import AbstractNeuron
from neurons.validators.penalty.penalty import BasePenaltyModel, PenaltyModelType

MAX_PENALTY = 1.0


def _relevance_value(value):
    if value is None:
        return None
    return value.value if isinstance(value, ContextualRelevance) else value


class MinerScorePenaltyModel(BasePenaltyModel):
    def __init__(self, max_penalty: float = MAX_PENALTY, neuron: AbstractNeuron = None):
        super().__init__(max_penalty, neuron)
        bt.logging.debug(
            "Initialized MinerScorePenaltyModel using max_execution_time from responses."
        )

    @property
    def name(self) -> str:
        return PenaltyModelType.miner_score_penalty.value

    async def calculate_penalties(
        self,
        responses: List[ScraperStreamingSynapse],
        additional_params=None,
    ) -> np.ndarray:
        penalties = np.zeros(len(responses), dtype=np.float32)

        for index, response in enumerate(responses):
            val_scores = [scores[index] for scores in additional_params]

            scores = []
            for score in val_scores:
                for link, validator_label in score.items():
                    miner_label = _relevance_value(response.miner_link_scores.get(link))
                    scores.append(1 if miner_label == validator_label else 0)

            score = sum(scores) / len(scores) if scores else 1
            penalties[index] = 1 - score

        return penalties
