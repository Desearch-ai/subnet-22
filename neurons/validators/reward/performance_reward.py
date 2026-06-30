import traceback
from typing import List, Tuple

import bittensor as bt

from desearch.protocol import (
    ResultType,
    ScraperStreamingSynapse,
    TwitterIDSearchSynapse,
    TwitterSearchSynapse,
    TwitterURLsSearchSynapse,
    WebSearchSynapse,
)
from neurons.validators.base_validator import AbstractNeuron

from .config import RewardModelType
from .reward import BaseRewardEvent, BaseRewardModel, log_reward_aggregates


AI_PERF_FLOOR = 0.50
WEB_PERF_FLOOR = 0.70
X_PERF_FLOOR = 0.70


def perf_factor(perf_raw: float, floor: float) -> float:
    return floor + (1.0 - floor) * perf_raw


def resolve_scoring_budget(response) -> float:
    from desearch.utils import get_mode_budget

    mode = getattr(response, "mode", None)
    if mode:
        try:
            return float(get_mode_budget(mode))
        except (KeyError, ValueError):
            pass
    raw = getattr(response, "max_execution_time", None)
    try:
        return float(raw) if raw is not None else 0.0
    except (TypeError, ValueError):
        return 0.0


def min_realistic_for_budget(budget: float, default: float) -> float:
    if not budget or budget <= 0:
        return default
    return min(2.0, 0.3 * budget)


class PerformanceRewardModel(BaseRewardModel):
    is_deep = False

    @property
    def name(self) -> str:
        return RewardModelType.performance_score.value

    def __init__(
        self,
        neuron: AbstractNeuron,
        min_realistic_time: float,
        target_time: float,
    ):
        super().__init__(neuron)
        self.min_realistic_time = min_realistic_time
        self.target_time = target_time

    def get_successful_streaming_response(self, response: ScraperStreamingSynapse):
        if response.result_type == ResultType.ONLY_LINKS:
            return self.get_successful_twitter_completion(
                response
            ) or self.get_successful_search_summary_completion(response)

        return self.get_successful_completion(response)

    def get_response_times(
        self, uids: List[int], responses: List[ScraperStreamingSynapse]
    ) -> List[float]:
        """
        Returns response times aligned by response index.
        Failed or unsuccessful completions are pinned to max_execution_time so the
        piecewise curve resolves them to reward 0.
        """
        response_times = []
        for response in responses:
            successful_response = self.get_successful_streaming_response(response)
            if response.dendrite.process_time is not None and successful_response:
                response_times.append(response.dendrite.process_time)
            else:
                response_times.append(response.max_execution_time)
        return response_times

    def get_global_response_times(
        self, uids: List[int], responses: List[TwitterSearchSynapse]
    ) -> List[float]:
        """
        Returns response times aligned by response index for global results.
        Empty or invalid results are pinned to max_execution_time (reward 0).
        Previously these were pinned to 0.0, which let instant empty responses game
        the sigmoid into near-max reward.
        """
        response_times = []
        for idx, response in enumerate(responses):
            uid = uids[idx]
            successful_result = self.get_successful_result(response)

            if successful_result:
                response_times.append(response.dendrite.process_time or 0.0)
            else:
                bt.logging.warning(
                    f"Invalid or empty result for UID: {uid}, pinning to timeout."
                )
                response_times.append(response.max_execution_time)

        return response_times

    def _thresholds_for(self, budget: float) -> Tuple[float, float]:
        if not budget or budget <= 0:
            return self.min_realistic_time, self.target_time
        return min_realistic_for_budget(budget, self.min_realistic_time), 0.6 * budget

    def _scoring_budget(self, response) -> float:
        return resolve_scoring_budget(response)

    def reward(self, axon_time: float, budget: float) -> float:
        min_realistic, target = self._thresholds_for(budget)

        if axon_time < min_realistic:
            return 0.0
        if axon_time <= target:
            return 1.0
        if budget and budget > 0:
            if axon_time <= budget:
                return 1.0 - 0.5 * (axon_time - target) / (budget - target)
            over = 0.5 * budget
            if axon_time <= budget + over:
                return 0.5 * (1.0 - (axon_time - budget) / over)
        return 0.0

    async def get_rewards(self, responses: List, uids) -> Tuple[List[BaseRewardEvent]]:
        """
        Returns a list of reward events for the given responses.
        """
        reward_events = []
        try:
            uids = [uid.item() if hasattr(uid, "item") else uid for uid in uids]

            if isinstance(responses[0], ScraperStreamingSynapse):
                response_times = self.get_response_times(uids, responses)
            elif isinstance(
                responses[0],
                (
                    TwitterSearchSynapse,
                    TwitterIDSearchSynapse,
                    TwitterURLsSearchSynapse,
                    WebSearchSynapse,
                ),
            ):
                response_times = self.get_global_response_times(uids, responses)
            else:
                raise ValueError("Unsupported response type provided to get_rewards.")

            for response_time, response in zip(response_times, responses):
                reward_event = BaseRewardEvent()
                reward_event.reward = self.reward(
                    response_time, self._scoring_budget(response)
                )
                reward_events.append(reward_event)

            log_reward_aggregates(
                name=self.name,
                uids=uids,
                scores=[event.reward for event in reward_events],
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
