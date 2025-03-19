# The MIT License (MIT)
# Copyright © 2023 Yuma Rao
# Copyright © 2023 Opentensor Foundation

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
import torch
import bittensor as bt
from enum import Enum
from typing import List
from abc import ABC, abstractmethod
from neurons.validators.utils.tasks import Task


class BasePenaltyModel(ABC):
    def __init__(self, max_penalty: float):
        self.max_penalty = max_penalty

    @property
    @abstractmethod
    def name(self) -> str: ...

    def __str__(self) -> str:
        return str(self.name)

    def __repr__(self) -> str:
        return str(self.name)

    @abstractmethod
    async def calculate_penalties(
        responses: List[bt.Synapse], tasks: List[Task], additional_params=None
    ) -> torch.FloatTensor: ...

    async def apply_penalties(
        self, responses: List[bt.Synapse], tasks: List[Task], additional_params=None
    ) -> torch.FloatTensor:
        raw_penalties = await self.calculate_penalties(
            responses, tasks, additional_params
        )

        # Clip penalties between 0 and 1
        adjusted_penalties = torch.clip(raw_penalties, 0, 1)

        # Clip penalties between 0 and self.max_penalty
        adjusted_penalties = torch.clip(adjusted_penalties, 0, self.max_penalty)

        # Invert penalties to scale rewards accordingly
        applied_penalties = 1 - adjusted_penalties

        return raw_penalties, adjusted_penalties, applied_penalties


class PenaltyModelType(Enum):
    task_validation_penalty = "task_validation_penalty"
    accuracy_match_penalty = "accuracy_match_penalty"
    link_validation_penalty = "link_validation_penalty"
    streaming_penalty = "streaming_penalty"
    exponential_penalty = "exponential_penalty"
    summary_rule_penalty = "summary_rule_penalty"
    twitter_count_penalty = "twitter_count_penalty"
    miner_score_penalty = "miner_score_penalty"
