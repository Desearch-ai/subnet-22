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

import asyncio
import re
from abc import abstractmethod
from dataclasses import asdict, dataclass, fields
from itertools import islice
from typing import List, Union

import bittensor as bt
import numpy as np  # Ensure numpy is imported
import torch

from desearch.protocol import (
    ScraperStreamingSynapse,
    TwitterSearchSynapse,
)
from neurons.validators.base_validator import AbstractNeuron


@dataclass
class BaseRewardEvent:
    reward: float = 1.0
    normalized_reward: float = None

    @staticmethod
    def parse_reward_events(reward_events):
        if reward_events == None or len(reward_events) == 0:
            field_names = [field.name for field in fields(BaseRewardEvent())]
            empty_reward_event = dict(zip(field_names, [[]] * len(field_names)))
            return empty_reward_event

        field_names = [field.name for field in fields(reward_events[0])]
        reward_events = [
            asdict(reward_event).values() for reward_event in reward_events
        ]
        reward_event = dict(zip(field_names, list(zip(*reward_events))))
        return reward_event


pattern_to_check = r"<(?:Question|/Question|Answer|/Answer|Score|/Score)>|SM(?:[-_ ]SCS)?[-_ ]?(?:RDD|PNK|BLE|GRY|GRN)"


class BaseRewardModel:
    @property
    @abstractmethod
    def name(self) -> str: ...

    def __str__(self) -> str:
        return str(self.name)

    def __repr__(self) -> str:
        return str(self.name)

    @abstractmethod
    async def get_rewards(
        self, responses: List[ScraperStreamingSynapse], name: str, uids
    ) -> Union[torch.FloatTensor, dict]: ...

    def __init__(self, neuron: AbstractNeuron) -> None:
        self.count = 0
        self.mean = 0.0
        self.var = 0.0
        self.is_default_normalization = True
        self.neuron = neuron

    def normalize_rewards(self, rewards: torch.FloatTensor) -> torch.FloatTensor:
        # if self.var > 0:
        #     rewards /= torch.sqrt(self.var)

        # Identify rewards that are initially 0
        zero_mask = rewards == 0

        # Min-max normalize rewards to scale between 0 and 1
        min_reward = (
            torch.min(rewards[~zero_mask]) if rewards[~zero_mask].nelement() > 0 else 0
        )
        max_reward = (
            torch.max(rewards[~zero_mask]) if rewards[~zero_mask].nelement() > 0 else 1
        )

        # Check if all non-zero rewards are the same
        if min_reward == max_reward:
            rewards[~zero_mask] = 1  # Set all non-zero rewards to 1
        else:
            epsilon = 1e-10
            rewards[~zero_mask] = (rewards[~zero_mask] - min_reward) / (
                max(max_reward - min_reward, epsilon)
            )

            # Apply a more aggressive exponential function to exaggerate differences
            exaggeration_factor = 4
            rewards[~zero_mask] = torch.pow(
                rewards[~zero_mask] * exaggeration_factor, exaggeration_factor
            )

            # Re-scale to ensure the top score is close to 1 after exponential exaggeration
            if rewards[~zero_mask].nelement() > 0:
                rewards[~zero_mask] /= torch.max(rewards[~zero_mask])

        # Ensure rewards that were initially 0 remain 0
        rewards[zero_mask] = 0

        return rewards

    def validate_successful_completion(self, response, completion: str):
        if response.dendrite.status_code == 200 and completion:
            if re.search(pattern_to_check, completion, flags=re.IGNORECASE):
                bt.logging.info(
                    f"Pattern validation issue Hotkey ID: {response.axon.hotkey}."
                )
                return None

            return completion.strip()

    def get_successful_completion(self, response: ScraperStreamingSynapse):
        # Check if the response is successful.
        if response.dendrite.status_code == 200:
            # Get the completion from the successful response.
            successful_completion = response.completion.strip()

            if re.search(pattern_to_check, successful_completion, flags=re.IGNORECASE):
                bt.logging.info(
                    f"Pattern validation issue Hotkey ID: {response.axon.hotkey}."
                )
                return None

            return successful_completion.strip()
        return None

    def get_successful_result(self, response: TwitterSearchSynapse):
        """
        Check if the response is successful and contains non-empty results.
        """
        if response.dendrite.status_code == 200:
            # Ensure results is not empty
            if response.results:
                return response.results
            else:
                bt.logging.warning(
                    f"Response results are empty for Hotkey ID: {response.axon.hotkey}."
                )
        else:
            bt.logging.warning(
                f"Response failed with status code {response.dendrite.status_code} for Hotkey ID: {response.axon.hotkey}."
            )

        return None

    def get_successful_completions(self, responses: List[ScraperStreamingSynapse]):
        successful_completions = [
            self.get_successful_completion(response) for response in responses
        ]
        return [
            completion
            for completion in successful_completions
            if completion is not None
        ]

    def get_successful_twitter_completion(self, response: ScraperStreamingSynapse):
        # Check if the response is successful.
        if response.dendrite.status_code == 200 and response.miner_tweets:
            return True

        return None

    def get_successful_search_summary_completion(
        self, response: ScraperStreamingSynapse
    ):
        # Check if the response is successful.
        links, _ = response.get_links_from_search_results()

        if response.dendrite.status_code == 200 and links:
            return True

        return None

    def get_successful_search_completions(
        self, responses: List[ScraperStreamingSynapse]
    ):
        successful_completions = [
            self.get_successful_search_summary_completion(response)
            for response in responses
        ]
        return [
            completion
            for completion in successful_completions
            if completion is not None
        ]

    async def apply(
        self,
        responses: List[ScraperStreamingSynapse],
        uids,
    ) -> Union[torch.FloatTensor, dict]:
        """Applies the reward model across each call. Unsuccessful responses are zeroed."""
        # Get indices of correctly responding calls.

        successful_completions_indices: List[int] = [
            idx
            for idx, resp in enumerate(responses)
            if resp.dendrite.status_code == 200
        ]

        reward_events, val_score_responses = await self.get_rewards(responses, uids)

        # Reward each completion.
        reward_events = BaseRewardEvent.parse_reward_events(reward_events)
        successful_rewards = reward_events
        successful_rewards = torch.tensor(
            reward_events.pop("reward"), dtype=torch.float32
        )

        original_rewards = successful_rewards.tolist()

        # Softmax rewards across samples.
        if self.is_default_normalization:
            successful_rewards_normalized = self.normalize_rewards(successful_rewards)
        else:
            successful_rewards_normalized = original_rewards

        # Init zero rewards for all calls.
        filled_rewards = torch.ones(len(responses), dtype=torch.float32) * torch.nan
        filled_rewards_normalized = torch.zeros(len(responses), dtype=torch.float32)

        for idx in successful_completions_indices:
            filled_rewards[idx] = successful_rewards[idx]
            filled_rewards_normalized[idx] = successful_rewards_normalized[idx]

        # Fill every item of the reward_events
        for name, reward_values in reward_events.items():
            filled_values = [None] * len(responses)
            for idx, reward_value in zip(successful_completions_indices, reward_values):
                filled_values[idx] = reward_value
            reward_events[name] = filled_values

        # Name each item of the reward event with the reward model name.
        reward_events = {f"{self.name}_{k}": v for k, v in reward_events.items()}
        reward_events[self.name] = filled_rewards.tolist()
        reward_events[self.name + "_normalized"] = filled_rewards_normalized.tolist()

        # Warns unexpected behavior for rewards
        if torch.isnan(filled_rewards_normalized).any():
            bt.logging.warning(
                f"The tensor from {self.name} contains NaN values: {filled_rewards_normalized}"
            )
            filled_rewards_normalized = filled_rewards_normalized.nan_to_num_(nan=0.0)

        # Return the filled rewards.
        return (
            filled_rewards_normalized,
            reward_events,
            val_score_responses,
            original_rewards,
        )

    def calculate_adjusted_score(
        self,
        links_count: int,
        score: float,
        duplicate_tweets_count: int = 0,
        max_bonus: float = 0.2,
        link_sensitivity: int = 9,
        max_links_threshold: int = 10,
        penalty_factor: float = 0.1,
    ) -> float:
        """
        Calculate the combined score by first applying a bonus based on the number of links and then adjusting
        the score based on the number of completion links with a softer penalty for having fewer than 10 links.
        If the number of links exceeds the max_links_threshold, a penalty is applied.

        Args:
        - score (float): The original score ranging from 0.1 to 1.
        - links_count (int): The number of links or completion links.
        - max_bonus (float): The maximum bonus to add to the score for the link count scenario. Default is 0.2.
        - link_sensitivity (int): Controls how quickly the bonus grows with the number of links. Higher values mean slower growth.
        - max_links_threshold (int): The threshold for the maximum number of links before penalties apply.
        - penalty_factor (float): The penalty applied for each link above the threshold.

        Returns:
        - float: The combined adjusted score considering the provided parameters.
        """
        # Calculate the bonus based on the number of links
        bonus = max_bonus * (
            1 - 1 / (1 + min(links_count, max_links_threshold) / link_sensitivity)
        )
        intermediate_score = min(1, score + bonus)

        # Adjust the intermediate score based on the number of completion links
        if links_count <= max_links_threshold:
            # Using square root to soften the penalty for having fewer than max_links_threshold links
            penalty_factor = (links_count / max_links_threshold) ** 0.5
        else:
            # Apply a penalty for each link over the threshold
            excess_links = links_count - max_links_threshold
            penalty_factor = max(0, 1 - excess_links * penalty_factor)

        adjusted_score = intermediate_score * penalty_factor

        if duplicate_tweets_count > 0:
            penalty_score = duplicate_tweets_count * 0.05
            adjusted_score = max(0, adjusted_score - penalty_score)

        return adjusted_score

    async def process_response_items_in_batches(
        self, responses, batch_size, process_function
    ):
        """Process validator links or tweets in sequence groups to avoid OpenAI timeouts."""
        results = []

        # Helper function to split items into chunks of batch_size
        def chunked(iterable, size):
            iterator = iter(iterable)
            for first in iterator:
                yield [first] + list(islice(iterator, size - 1))

        # Process items in batches
        for batch in chunked(responses, batch_size):
            batch_results = await asyncio.gather(
                *[process_function(response) for response in batch]
            )
            results.extend(batch_results)
        return results
