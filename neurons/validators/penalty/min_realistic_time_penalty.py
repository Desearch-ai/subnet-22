from typing import List, Optional

import bittensor as bt
import numpy as np

from neurons.validators.base_validator import AbstractNeuron
from neurons.validators.penalty.penalty import BasePenaltyModel, PenaltyModelType

MAX_PENALTY = 1.0


class MinRealisticTimePenaltyModel(BasePenaltyModel):
    """Penalize responses returned faster than ``min_realistic_time``. A miner
    that returns well-formed content in under-realistic time is almost
    certainly serving cached data rather than running the requested search."""

    is_deep = False

    def __init__(
        self,
        min_realistic_time: float,
        max_penalty: float = MAX_PENALTY,
        neuron: AbstractNeuron = None,
    ):
        super().__init__(max_penalty, neuron)
        self.min_realistic_time = min_realistic_time

    @property
    def name(self) -> str:
        return PenaltyModelType.min_realistic_time_penalty.value

    @staticmethod
    def _safe_float(value) -> Optional[float]:
        if value is None or value == "":
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    async def calculate_penalties(
        self,
        responses: List[bt.Synapse],
        additional_params=None,
    ) -> np.ndarray:
        penalties = np.zeros(len(responses), dtype=np.float32)
        for i, response in enumerate(responses):
            dendrite = getattr(response, "dendrite", None)
            process_time = self._safe_float(getattr(dendrite, "process_time", None))
            if process_time is None:
                continue
            if process_time < self.min_realistic_time:
                penalties[i] = self.max_penalty
        return penalties
