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

import traceback
import torch
import bittensor as bt
from typing import List, Tuple, Dict
import math
import json
from .config import RewardModelType
from .reward import BaseRewardModel, BaseRewardEvent
from datura.protocol import (
    ScraperStreamingSynapse,
    TwitterSearchSynapse,
    TwitterIDSearchSynapse,
    TwitterURLsSearchSynapse,
    WebSearchSynapse,
    PeopleSearchSynapse,
)
from neurons.validators.constants import STEEPNESS, FACTOR


class PerformanceRewardModel(BaseRewardModel):
    @property
    def name(self) -> str:
        return RewardModelType.performance_score.value

    def __init__(self, device: str):
        super().__init__()
        self.device = device
        self.is_default_normalization = False

    def get_response_times(
        self, uids: List[int], responses: List[ScraperStreamingSynapse]
    ) -> Dict[int, float]:
        """
        Returns a dictionary of axons based on their response times.
        Adds a check for successful completion of the response.
        """
        axon_times = {
            uids[idx]: (
                response.dendrite.process_time
                if response.dendrite.process_time is not None
                and self.get_successful_completion(response)
                else response.max_execution_time
            )
            for idx, response in enumerate(responses)
        }
        return axon_times

    def get_global_response_times(
        self, uids: List[int], responses: List[TwitterSearchSynapse]
    ) -> Dict[int, float]:
        """
        Returns a dictionary of axons based on their response times for global results.
        If get_successful_result returns an invalid result (e.g., an empty list), the score is 0.
        """
        axon_times = {}
        for idx, response in enumerate(responses):
            uid = uids[idx]
            successful_result = self.get_successful_result(response)

            if successful_result:  # If successful_result is valid and non-empty
                axon_times[uid] = response.dendrite.process_time or 0.0
            else:  # Invalid result or empty list
                bt.logging.warning(
                    f"Invalid or empty result for UID: {uid}, setting score to 0."
                )
                axon_times[uid] = 0.0

        return axon_times

    def sigmoid_scale(self, axon_time: float, query_timeout: int) -> float:
        """
        Scales the axon time using a sigmoid function.
        """
        offset = -10.0 / FACTOR
        return (
            (1 / (1 + math.exp(STEEPNESS * axon_time + offset)))
            if axon_time < query_timeout
            else 0
        )

    def reward(self, axon_time: float, query_timeout: int) -> float:
        """
        Calculates the reward for a miner based on axon time and APY.
        """
        return 0.2 * self.sigmoid_scale(axon_time, query_timeout)

    async def get_rewards(self, responses: List, uids) -> Tuple[List[BaseRewardEvent]]:
        """
        Returns a list of reward events for the given responses.
        """
        reward_events = []
        try:
            # Convert tensor uids to integers if necessary
            uids = [
                uid.item() if isinstance(uid, torch.Tensor) else uid for uid in uids
            ]

            # Determine response type and select the appropriate response time function
            if isinstance(responses[0], ScraperStreamingSynapse):
                axon_times = self.get_response_times(uids, responses)
            elif isinstance(
                responses[0],
                (
                    TwitterSearchSynapse,
                    TwitterIDSearchSynapse,
                    TwitterURLsSearchSynapse,
                    WebSearchSynapse,
                    PeopleSearchSynapse,
                ),
            ):
                axon_times = self.get_global_response_times(uids, responses)
            else:
                raise ValueError("Unsupported response type provided to get_rewards.")

            for uid, response in zip(uids, responses):
                reward_event = BaseRewardEvent()
                reward_event.reward = self.reward(
                    axon_times[uid], response.max_execution_time
                )
                reward_events.append(reward_event)

            zero_rewards = [event for event in reward_events if event.reward == 0]
            non_zero_rewards = [event for event in reward_events if event.reward != 0]

            bt.logging.info(
                f"==================================Performance Reward Zero Rewards ({len(zero_rewards)} cases)=================================="
            )
            bt.logging.info(json.dumps([event.reward for event in zero_rewards]))
            bt.logging.info(
                f"==================================Performance Reward Non-Zero Rewards ({len(non_zero_rewards)} cases)=================================="
            )
            bt.logging.info(
                json.dumps([round(event.reward, 6) for event in non_zero_rewards])
            )
            return reward_events, {}
        except Exception as e:
            error_message = f"PerformanceRewardModel get_rewards: {str(e)}"
            tb_str = traceback.format_exception(type(e), e, e.__traceback__)
            bt.logging.error("\n".join(tb_str) + error_message)
            for uid in uids:
                reward_event = BaseRewardEvent()
                reward_event.reward = 0
                reward_events.append(reward_event)
            return reward_events, {}
