from typing import Any, Dict, List, Optional

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

        self.web_content_weight = 0.80
        self.performance_weight = 0.20

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

    async def call_miner(
        self,
        prompt: str,
        params: Dict[str, Any],
        uid: Optional[int] = None,
    ):
        uid, axon = await self.neuron.get_random_miner(
            uid=uid, search_type=self.search_type
        )
        worker_url = self.neuron.miner_worker_urls.get(uid)

        synapse = WebSearchSynapse(
            **params,
            query=prompt,
            max_execution_time=self.max_execution_time,
        )

        response = await self.neuron.worker_client.call_json_search(
            worker_url,
            "/web/search",
            synapse.model_copy(),
            WebSearchSynapse,
            axon,
            uid=uid,
            search_type=self.search_type,
        )

        return response, uid, axon

    async def send_scoring_query(
        self,
        query: dict,
        uid: int,
    ) -> Optional[object]:
        """
        Send a scoring query to a specific miner via worker URL.
        Called by QueryScheduler; returns the fully-populated synapse.
        """
        prompt = query.get("query", "")
        params = {k: v for k, v in query.items() if k != "query"}

        worker_url = self.neuron.miner_worker_urls.get(uid)
        if not worker_url:
            bt.logging.warning(f"[Web] No worker_url for uid={uid}, skipping")
            return None

        synapse = WebSearchSynapse(
            **params,
            query=prompt,
            max_execution_time=self.max_execution_time,
        )

        axon = self.neuron.metagraph.axons[uid]
        response = await self.neuron.worker_client.call_json_search(
            worker_url,
            "/web/search",
            synapse,
            WebSearchSynapse,
            axon,
            uid=uid,
            search_type=self.search_type,
        )
        return response

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
