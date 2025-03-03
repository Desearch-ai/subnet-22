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
from typing import Union, List
from neurons.validators.reward import BaseRewardModel, BaseRewardEvent
from datura.protocol import ScraperStreamingSynapse


class MockRewardModel(BaseRewardModel):
    question_blacklist = []
    answer_blacklist = []

    @property
    def name(self) -> str:
        return self.mock_name

    def __init__(self, mock_name: str = "MockReward"):
        super().__init__()
        self.mock_name = mock_name
        self.question_blacklist = []
        self.answer_blacklist = []

    def add(self, texts: List[str]):
        pass

    def set_counter_to_half(self):
        pass

    async def apply(
        self,
        responses: List[ScraperStreamingSynapse],
        uids,
        organic_penalties: List[bool] = [],
    ) -> Union[torch.FloatTensor, dict]:
        reward_normalized = torch.tensor([1 for _ in responses], dtype=torch.float32)
        reward_events = [BaseRewardEvent(reward=1) for _ in responses]
        grouped_val_score_responses = [[{}] for _ in responses]
        original_rewards = torch.tensor([1 for _ in responses], dtype=torch.float32)

        return (
            reward_normalized,
            reward_events,
            grouped_val_score_responses,
            original_rewards,
        )

    def reset(self):
        return self

    def reward(
        self,
        completions_with_prompt: List[str],
        completions_without_prompt: List[str],
        difference=False,
        shift=3,
    ) -> torch.FloatTensor:
        return torch.zeros(len(completions_with_prompt))
