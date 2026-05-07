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


def score_to_contextual_relevance(score):
    if score == 2.0:
        return ContextualRelevance.LOW
    elif score == 5.0:
        return ContextualRelevance.MEDIUM
    elif score == 9.0:
        return ContextualRelevance.HIGH


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
                for link, link_score in score.items():
                    if response.miner_link_scores.get(
                        link
                    ) != score_to_contextual_relevance(link_score):
                        scores.append(0)
                    else:
                        scores.append(1)

            score = sum(scores) / len(scores) if scores else 1
            penalties[index] = 1 - score

            bt.logging.debug(f"Response index {index} has penalty {penalties[index]}")

        return penalties
