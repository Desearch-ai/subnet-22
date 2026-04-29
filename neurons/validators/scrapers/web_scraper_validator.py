from typing import Any, Dict, Optional

import bittensor as bt
import torch

from desearch.protocol import (
    WebSearchSynapse,
)
from neurons.validators.base_validator import AbstractNeuron
from neurons.validators.clients.miner_response_logger import (
    build_log_entry,
    submit_logs_best_effort,
)
from neurons.validators.penalty.timeout_penalty import TimeoutPenaltyModel
from neurons.validators.reward import RewardScoringType
from neurons.validators.reward.performance_reward import PerformanceRewardModel
from neurons.validators.reward.web_basic_search_content_relevance import (
    WebBasicSearchContentRelevanceModel,
)
from neurons.validators.scrapers.base_scraper_validator import BaseScraperValidator


class WebScraperValidator(BaseScraperValidator):
    search_type = "web_search"
    wandb_modality = "web_scrapper"
    wandb_reward_keys = ["search_reward"]

    def __init__(self, neuron: AbstractNeuron):
        self.timeout = 180
        self.max_execution_time = 10

        # Init device.
        bt.logging.debug("loading", "device")
        bt.logging.debug(
            "self.neuron.config.neuron.device = ", str(neuron.config.neuron.device)
        )

        self.web_content_weight = 0.70
        self.performance_weight = 0.30

        reward_weights = torch.tensor(
            [
                self.web_content_weight,
                self.performance_weight,
            ],
            dtype=torch.float32,
        )

        reward_functions = [
            WebBasicSearchContentRelevanceModel(
                device=neuron.config.neuron.device,
                scoring_type=RewardScoringType.search_relevance_score_template,
                neuron=neuron,
            ),
            PerformanceRewardModel(
                device=neuron.config.neuron.device,
                neuron=neuron,
                min_realistic_time=1.0,
                target_time=3.0,
            ),
        ]

        penalty_functions = [
            TimeoutPenaltyModel(max_penalty=1, neuron=neuron),
        ]

        super().__init__(
            neuron=neuron,
            reward_weights=reward_weights,
            reward_functions=reward_functions,
            penalty_functions=penalty_functions,
        )

    def _build_synapse(
        self, prompt: str, params: Dict[str, Any]
    ) -> WebSearchSynapse:
        return WebSearchSynapse(
            **params,
            query=prompt,
            max_execution_time=self.max_execution_time,
        )

    async def call_miner(
        self,
        prompt: str,
        params: Dict[str, Any],
        uid: Optional[int] = None,
    ):
        uid, axon = await self.neuron.get_random_miner(
            uid=uid, search_type=self.search_type
        )
        synapse = self._build_synapse(prompt, params)
        response = await self._dendrite_call(axon, synapse.model_copy(), uid)
        return response, uid, axon

    async def send_scoring_query(
        self,
        query: dict,
        uid: int,
    ) -> Optional[object]:
        """Send a scoring query to a specific miner via dendrite.
        Called by QueryScheduler; returns the fully-populated synapse."""
        prompt = query.get("query", "")
        params = {k: v for k, v in query.items() if k != "query"}

        synapse = self._build_synapse(prompt, params)
        axon = self.neuron.metagraph.axons[uid]
        return await self._dendrite_call(axon, synapse, uid)

    async def organic(
        self,
        query,
    ):
        """Receives question from user and returns the response from the miners."""

        try:
            prompt = query.get("query", "")
            params = {key: value for key, value in query.items() if key != "query"}

            response, selected_uid, axon = await self.call_miner(
                prompt=prompt, params=params
            )

            if response:
                submit_logs_best_effort(
                    self.neuron,
                    [
                        build_log_entry(
                            owner=self.neuron,
                            search_type="web_search",
                            query_kind="organic",
                            response=response,
                            miner_uid=selected_uid,
                            miner_hotkey=getattr(axon, "hotkey", None),
                            miner_coldkey=getattr(axon, "coldkey", None),
                        )
                    ],
                )
                await self._save_organic_for_scoring(
                    uid=selected_uid, response=response
                )
                yield response
            else:
                bt.logging.warning("Invalid response for UID: Unknown")

        except Exception as e:
            bt.logging.error(f"Error in organic: {e}")
            raise e
