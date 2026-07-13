from typing import Optional

from neurons.validators.base_validator import AbstractNeuron
from neurons.validators.penalty.penalty import CheapPenaltyModel, PenaltyModelType
from neurons.validators.reward.performance_reward import (
    min_realistic_for_budget,
    resolve_scoring_budget,
)


class MinRealisticTimePenaltyModel(CheapPenaltyModel):
    """Penalize responses returned faster than the realistic time for their
    mode budget (almost certainly cached, not a real search)."""

    name = PenaltyModelType.min_realistic_time_penalty.value

    def __init__(self, max_penalty: float = 1.0, neuron: AbstractNeuron = None):
        super().__init__(max_penalty, neuron)

    @staticmethod
    def _safe_float(value) -> Optional[float]:
        if value is None or value == "":
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def _min_realistic_for(self, response) -> float:
        return min_realistic_for_budget(resolve_scoring_budget(response))

    def penalty_for(self, response) -> float:
        dendrite = getattr(response, "dendrite", None)
        process_time = self._safe_float(getattr(dendrite, "process_time", None))
        if process_time is None:
            return 0.0

        if process_time < self._min_realistic_for(response):
            return self.max_penalty
        return 0.0
