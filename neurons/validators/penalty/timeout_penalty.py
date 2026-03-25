import math
from typing import List, Optional

import bittensor as bt
import torch

from neurons.validators.base_validator import AbstractNeuron
from neurons.validators.penalty.penalty import BasePenaltyModel, PenaltyModelType

MAX_PENALTY = 1.0
DEFAULT_TIMEOUT_GRACE_SECONDS = 5.0


class TimeoutPenaltyModel(BasePenaltyModel):
    def __init__(
        self,
        max_penalty: float = MAX_PENALTY,
        neuron: AbstractNeuron = None,
        timeout_grace_seconds: float = DEFAULT_TIMEOUT_GRACE_SECONDS,
    ):
        super().__init__(max_penalty, neuron)
        self.timeout_grace_seconds = timeout_grace_seconds
        bt.logging.debug(
            "Initialized TimeoutPenaltyModel using response timeouts when available."
        )

    @property
    def name(self) -> str:
        return PenaltyModelType.timeout_penalty.value

    @staticmethod
    def _safe_float(value) -> Optional[float]:
        if value is None or value == "":
            return None

        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def _get_timeout_window(self, response, max_execution_time: float) -> float:
        response_timeout = self._safe_float(getattr(response, "timeout", None))

        if response_timeout is None or response_timeout <= max_execution_time:
            return self.timeout_grace_seconds

        return response_timeout - max_execution_time

    async def calculate_penalties(
        self,
        responses: List[bt.Synapse],
        additional_params=None,
    ) -> torch.FloatTensor:
        penalties = torch.zeros(len(responses), dtype=torch.float32)

        for index, response in enumerate(responses):
            dendrite = getattr(response, "dendrite", None)
            process_time = self._safe_float(getattr(dendrite, "process_time", None))
            max_execution_time = self._safe_float(
                getattr(response, "max_execution_time", None)
            )

            if process_time is None or max_execution_time is None:
                penalties[index] = self.max_penalty
                continue

            if process_time <= max_execution_time:
                penalties[index] = 0.0
                continue

            delay = process_time - max_execution_time
            timeout_window = max(
                self._get_timeout_window(response, max_execution_time),
                1e-6,
            )
            elapsed_seconds = math.ceil(delay)
            penalty_step = self.max_penalty / timeout_window
            penalty = min(elapsed_seconds * penalty_step, self.max_penalty)
            penalties[index] = penalty

        bt.logging.info(f"Timeout Penalties: {penalties.tolist()}")

        return penalties
