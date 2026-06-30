from abc import ABC, abstractmethod
from collections import defaultdict
from enum import Enum
from typing import List

import bittensor as bt
import numpy as np

from neurons.validators.base_validator import AbstractNeuron


class BasePenaltyModel(ABC):
    is_deep: bool = True

    def __init__(
        self,
        max_penalty: float = 1.0,
        neuron: AbstractNeuron = None,
    ):
        self.max_penalty = max_penalty
        self.neuron = neuron

    @property
    @abstractmethod
    def name(self) -> str: ...

    def __str__(self) -> str:
        return str(self.name)

    def __repr__(self) -> str:
        return str(self.name)

    @abstractmethod
    async def calculate_penalties(
        responses: List[bt.Synapse], additional_params=None
    ) -> np.ndarray: ...

    def _log_triggers(self, uids, raw_penalties: np.ndarray) -> None:
        """Emit one line per scoring batch summarizing which UIDs were hit."""
        per_uid = defaultdict(list)
        for uid, penalty in zip(uids, raw_penalties.tolist()):
            uid_val = uid.item() if hasattr(uid, "item") else int(uid)
            per_uid[uid_val].append(penalty)

        triggered = {
            uid: vals for uid, vals in per_uid.items() if any(v > 0 for v in vals)
        }
        total = len(raw_penalties)
        triggered_count = sum(1 for v in raw_penalties.tolist() if v > 0)

        if not triggered:
            bt.logging.info(
                f"[Penalty {self.name}] no triggers (0 of {total} responses)"
            )
            return

        bt.logging.info(
            f"[Penalty {self.name}] triggered on {triggered_count} of {total} "
            f"responses across {len(triggered)} UIDs:"
        )
        for uid in sorted(triggered):
            vals = triggered[uid]
            hit = sum(1 for v in vals if v > 0)
            bt.logging.info(
                f"  UID {uid}: {hit}/{len(vals)} triggered "
                f"(max={max(vals):.2f}, mean={sum(vals) / len(vals):.2f})"
            )

    async def apply_penalties(
        self,
        responses: List[bt.Synapse],
        uids,
        additional_params=None,
    ) -> np.ndarray:
        raw_penalties = await self.calculate_penalties(responses, additional_params)
        self._log_triggers(uids, raw_penalties)

        adjusted_penalties = np.clip(raw_penalties, 0, 1)
        adjusted_penalties = np.clip(adjusted_penalties, 0, self.max_penalty)

        applied_penalties = 1 - adjusted_penalties

        return raw_penalties, adjusted_penalties, applied_penalties


class CheapPenaltyModel(BasePenaltyModel):
    """Per-response, sync, no-IO penalty. Subclasses set ``name`` and override
    ``penalty_for(response) -> float`` returning a value in ``[0, max_penalty]``."""

    is_deep = False
    name: str = ""

    @abstractmethod
    def penalty_for(self, response) -> float: ...

    async def calculate_penalties(
        self,
        responses: List[bt.Synapse],
        additional_params=None,
    ) -> np.ndarray:
        return np.array([self.penalty_for(r) for r in responses], dtype=np.float32)


class PenaltyModelType(Enum):
    streaming_penalty = "streaming_penalty"
    timeout_penalty = "timeout_penalty"
    summary_rule_penalty = "summary_rule_penalty"
    count_penalty = "count_penalty"
    miner_score_penalty = "miner_score_penalty"
    summary_structure_penalty = "summary_structure_penalty"
    date_range_penalty = "date_range_penalty"
    duplicate_results_penalty = "duplicate_results_penalty"
    result_schema_penalty = "result_schema_penalty"
    sort_order_penalty = "sort_order_penalty"
    min_realistic_time_penalty = "min_realistic_time_penalty"
    domain_filter_penalty = "domain_filter_penalty"
