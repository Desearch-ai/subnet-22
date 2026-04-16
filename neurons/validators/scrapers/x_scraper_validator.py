from typing import Any, Dict, List, Optional

import bittensor as bt
import torch

from desearch.protocol import (
    TwitterIDSearchSynapse,
    TwitterSearchSynapse,
    TwitterURLsSearchSynapse,
)
from neurons.validators.base_validator import AbstractNeuron
from neurons.validators.clients.miner_response_logger import (
    build_log_entry,
    submit_logs_best_effort,
)
from neurons.validators.penalty.timeout_penalty import TimeoutPenaltyModel
from neurons.validators.penalty.twitter_count_penalty import TwitterCountPenaltyModel
from neurons.validators.reward import RewardScoringType
from neurons.validators.reward.performance_reward import PerformanceRewardModel
from neurons.validators.reward.twitter_basic_search_content_relevance import (
    TwitterBasicSearchContentRelevanceModel,
)
from neurons.validators.scrapers.base_scraper_validator import BaseScraperValidator


class XScraperValidator(BaseScraperValidator):
    search_type = "x_search"
    wandb_modality = "twitter_scrapper"
    wandb_reward_keys = ["twitter_reward"]

    def __init__(self, neuron: AbstractNeuron):
        self.timeout = 180
        self.max_execution_time = 10

        # Init device.
        bt.logging.debug("loading", "device")
        bt.logging.debug(
            "self.neuron.config.neuron.device = ", str(neuron.config.neuron.device)
        )

        self.twitter_content_weight = 0.80
        self.performance_weight = 0.20

        reward_weights = torch.tensor(
            [
                self.twitter_content_weight,
                self.performance_weight,
            ],
            dtype=torch.float32,
        )

        reward_functions = [
            TwitterBasicSearchContentRelevanceModel(
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
            TwitterCountPenaltyModel(max_penalty=1, neuron=neuron),
        ]

        super().__init__(
            neuron=neuron,
            reward_weights=reward_weights,
            reward_functions=reward_functions,
            penalty_functions=penalty_functions,
        )

    def calc_max_execution_time(self, count):
        if not count or count <= 20:
            return self.max_execution_time

        return self.max_execution_time + int((count - 20) / 20) * 5

    async def call_miner(
        self,
        prompt: str,
        params: Dict[str, Any],
        uid: Optional[int] = None,
    ):
        uid, axon = await self.neuron.get_random_miner(
            uid=uid, search_type=self.search_type
        )

        synapse = TwitterSearchSynapse(
            **params,
            query=prompt,
            max_execution_time=self.calc_max_execution_time(params.get("count")),
        )

        dendrite = next(self.neuron.dendrites)

        response = await dendrite.call(
            target_axon=axon,
            synapse=synapse.model_copy(),
            timeout=synapse.max_execution_time + 5,
            deserialize=False,
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
            bt.logging.warning(f"[X] No worker_url for uid={uid}, skipping")
            return None

        synapse = TwitterSearchSynapse(
            **params,
            query=prompt,
            max_execution_time=self.calc_max_execution_time(params.get("count")),
        )

        axon = self.neuron.metagraph.axons[uid]
        response = await self.neuron.worker_client.call_json_search(
            worker_url, "/twitter/search", synapse, TwitterSearchSynapse, axon
        )
        return response

    async def x_search(
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
                self._save_organic_log(
                    response=response,
                    miner_uid=selected_uid,
                    axon=axon,
                    search_type="x_search",
                )
                yield response
            else:
                bt.logging.warning("Invalid response for UID: Unknown")

        except Exception as e:
            bt.logging.error(f"Error in organic: {e}")
            raise e

    async def x_post_by_id(
        self,
        tweet_id: str,
    ):
        """
        Perform a Twitter search using a specific tweet ID.
        """

        try:
            uid, axon = await self.neuron.get_random_miner(
                search_type=self.search_type
            )

            synapse = TwitterIDSearchSynapse(
                id=tweet_id,
                max_execution_time=self.max_execution_time,
                validator_tweets=[],
                results=[],
            )

            timeout = self.max_execution_time + 5

            dendrite = next(self.neuron.dendrites)

            synapse: TwitterIDSearchSynapse = await dendrite.call(
                target_axon=axon,
                synapse=synapse,
                timeout=timeout,
                deserialize=False,
            )

            self._save_organic_log(
                response=synapse,
                miner_uid=uid,
                axon=axon,
                search_type="x_post_by_id",
            )

            return synapse.results
        except Exception as e:
            bt.logging.error(f"Error in ID search: {e}")
            raise e

    async def x_posts_by_urls(
        self,
        urls: List[str],
    ):
        """
        Perform a Twitter search using multiple tweet URLs.
        """

        try:
            bt.logging.debug("run_task", "twitter urls search")

            uid, axon = await self.neuron.get_random_miner(
                search_type=self.search_type
            )

            synapse = TwitterURLsSearchSynapse(
                urls=urls,
                max_execution_time=self.calc_max_execution_time(len(urls)),
                validator_tweets=[],
                results=[],
            )

            timeout = synapse.max_execution_time + 5

            dendrite = next(self.neuron.dendrites)

            synapse: TwitterURLsSearchSynapse = await dendrite.call(
                target_axon=axon,
                synapse=synapse,
                timeout=timeout,
                deserialize=False,
            )

            self._save_organic_log(
                response=synapse,
                miner_uid=uid,
                axon=axon,
                search_type="x_posts_by_urls",
            )

            return synapse.results
        except Exception as e:
            bt.logging.error(f"Error in URLs search: {e}")
            raise e

    def _save_organic_log(
        self, response, miner_uid: int, axon, search_type: str
    ) -> None:
        submit_logs_best_effort(
            self.neuron,
            [
                build_log_entry(
                    owner=self.neuron,
                    search_type=search_type,
                    query_kind="organic",
                    response=response,
                    miner_uid=miner_uid,
                    miner_hotkey=getattr(axon, "hotkey", None),
                    miner_coldkey=getattr(axon, "coldkey", None),
                )
            ],
        )
