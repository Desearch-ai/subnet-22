import time
from typing import Any, Dict, List, Optional

import bittensor as bt
import torch

import wandb
from desearch.protocol import (
    TwitterIDSearchSynapse,
    TwitterSearchSynapse,
    TwitterURLsSearchSynapse,
)
from neurons.validators.base_validator import AbstractNeuron
from neurons.validators.miner_response_logger import (
    build_log_entry,
    build_reward_payload,
    submit_logs_best_effort,
)
from neurons.validators.penalty.exponential_penalty import ExponentialTimePenaltyModel
from neurons.validators.penalty.twitter_count_penalty import TwitterCountPenaltyModel
from neurons.validators.reward import RewardScoringType
from neurons.validators.reward.twitter_basic_search_content_relevance import (
    TwitterBasicSearchContentRelevanceModel,
)


class XScraperValidator:
    def __init__(self, neuron: AbstractNeuron):
        self.neuron = neuron
        self.timeout = 180
        self.max_execution_time = 10

        # Init device.
        bt.logging.debug("loading", "device")
        bt.logging.debug(
            "self.neuron.config.neuron.device = ", str(self.neuron.config.neuron.device)
        )

        self.twitter_content_weight = 1.0

        self.reward_weights = torch.tensor(
            [
                self.twitter_content_weight,
            ],
            dtype=torch.float32,
        ).to(self.neuron.config.neuron.device)

        if self.reward_weights.sum() != 1:
            message = (
                f"Reward function weights do not sum to 1 (Current sum: {self.reward_weights.sum()}.)"
                f"Check your reward config file at `reward/config.py` or ensure that all your cli reward flags sum to 1."
            )
            bt.logging.error(message)
            raise Exception(message)

        self.reward_functions = [
            TwitterBasicSearchContentRelevanceModel(
                device=self.neuron.config.neuron.device,
                scoring_type=RewardScoringType.search_relevance_score_template,
                neuron=self.neuron,
            ),
        ]

        self.penalty_functions = [
            ExponentialTimePenaltyModel(max_penalty=1, neuron=self.neuron),
            TwitterCountPenaltyModel(max_penalty=1, neuron=self.neuron),
        ]

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
        uid, axon = await self.neuron.get_random_miner(uid=uid)

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

    async def compute_rewards_and_penalties(
        self,
        event,
        prompts: List[str],
        responses,
        uids,
        start_time,
        scoring_epoch_start=None,
    ):
        try:
            if not len(uids):
                bt.logging.warning("No UIDs provided for logging event.")
                return

            bt.logging.info("Computing rewards and penalties")

            rewards = torch.zeros(len(responses), dtype=torch.float32).to(
                self.neuron.config.neuron.device
            )

            all_rewards = []
            all_original_rewards = []
            val_score_responses_list = []

            bt.logging.trace(f"Received responses: {responses}")

            for weight_i, reward_fn_i in zip(
                self.reward_weights, self.reward_functions
            ):
                start_time = time.time()
                (
                    reward_i_normalized,
                    reward_event,
                    val_score_responses,
                    original_rewards,
                ) = await reward_fn_i.apply(responses, uids)

                all_rewards.append(reward_i_normalized)
                all_original_rewards.append(original_rewards)
                val_score_responses_list.append(val_score_responses)

                rewards += weight_i * reward_i_normalized.to(
                    self.neuron.config.neuron.device
                )
                if not self.neuron.config.neuron.disable_log_rewards:
                    event = {**event, **reward_event}
                execution_time = time.time() - start_time
                bt.logging.trace(str(reward_fn_i.name), reward_i_normalized.tolist())
                bt.logging.info(
                    f"Applied reward function: {reward_fn_i.name} in {execution_time / 60:.2f} minutes"
                )

            for penalty_fn_i in self.penalty_functions:
                (
                    raw_penalty_i,
                    adjusted_penalty_i,
                    applied_penalty_i,
                ) = await penalty_fn_i.apply_penalties(responses, uids)
                penalty_start_time = time.time()
                rewards *= applied_penalty_i.to(self.neuron.config.neuron.device)
                penalty_execution_time = time.time() - penalty_start_time
                if not self.neuron.config.neuron.disable_log_rewards:
                    event[penalty_fn_i.name + "_raw"] = raw_penalty_i.tolist()
                    event[penalty_fn_i.name + "_adjusted"] = adjusted_penalty_i.tolist()
                    event[penalty_fn_i.name + "_applied"] = applied_penalty_i.tolist()
                bt.logging.trace(str(penalty_fn_i.name), applied_penalty_i.tolist())
                bt.logging.info(
                    f"Applied penalty function: {penalty_fn_i.name} in {penalty_execution_time:.2f} seconds"
                )

            await self.neuron.update_moving_averaged_scores(uids, rewards)
            self.log_event(prompts, event, start_time, uids, rewards)

            scores = torch.zeros(len(self.neuron.metagraph.hotkeys))
            uid_scores_dict = {}
            wandb_data = {
                "modality": "twitter_scrapper",
                "prompts": {},
                "responses": {},
                "scores": {},
                "timestamps": {},
                "twitter_reward": {},
            }
            bt.logging.info(
                f"======================== Reward ==========================="
            )
            # Initialize an empty list to accumulate log messages
            log_messages = []
            for uid_tensor, reward, response in zip(uids, rewards.tolist(), responses):
                uid = uid_tensor.item()

                # Accumulate log messages instead of logging them immediately
                log_messages.append(f"UID: {uid}, R: {round(reward, 3)}")

            # Log the accumulated messages in groups of three
            for i in range(0, len(log_messages), 3):
                bt.logging.info(" | ".join(log_messages[i : i + 3]))

            bt.logging.info(
                f"======================== Reward ==========================="
            )
            bt.logging.info(f"this is a all reward {all_rewards} ")

            twitter_rewards = all_rewards[0]
            zipped_rewards = zip(uids, rewards.tolist(), responses, twitter_rewards)

            for uid_tensor, reward, response, twitter_reward in zipped_rewards:
                uid = uid_tensor.item()  # Convert tensor to int
                uid_scores_dict[uid] = reward
                scores[uid] = reward  # Now 'uid' is an int, which is a valid key type
                wandb_data["scores"][uid] = reward
                if hasattr(response, "query"):
                    wandb_data["prompts"][uid] = response.query
                elif hasattr(response, "id"):
                    wandb_data["prompts"][uid] = response.id
                elif hasattr(response, "urls"):
                    wandb_data["prompts"][uid] = response.urls
                wandb_data["twitter_reward"][uid] = twitter_reward

            if self.neuron.config.wandb_on:
                wandb.log(wandb_data)

            scoring_logs = []
            response_count = len(responses)

            for index, (uid_tensor, response, reward) in enumerate(
                zip(uids, responses, rewards.tolist())
            ):
                uid = uid_tensor.item()
                reward_payload = build_reward_payload(
                    search_type="x_search",
                    response_count=response_count,
                    index=index,
                    uid=uid,
                    total_reward=reward,
                    all_rewards=all_rewards,
                    all_original_rewards=all_original_rewards,
                    validator_scores=val_score_responses_list,
                    event=event,
                )
                scoring_logs.append(
                    build_log_entry(
                        owner=self.neuron,
                        search_type="x_search",
                        query_kind="scoring",
                        response=response,
                        miner_uid=uid,
                        total_reward=reward,
                        reward_payload=reward_payload,
                        scoring_epoch_start=scoring_epoch_start,
                    )
                )

            submit_logs_best_effort(self.neuron, scoring_logs)

            return rewards, uids, val_score_responses_list, event, all_original_rewards
        except Exception as e:
            bt.logging.error(f"Error in compute_rewards_and_penalties: {e}")
            raise e

    def log_event(self, prompts: List[str], event, start_time, uids, rewards):
        event.update(
            {
                "step_length": time.time() - start_time,
                "prompts": prompts,
                "uids": uids.tolist(),
                "rewards": rewards.tolist(),
            }
        )

        bt.logging.debug("Run Task event:", event)

    async def send_scoring_query(
        self,
        query: dict,
        uid: int,
    ) -> Optional[object]:
        """
        Send a scoring query to a specific miner and return the full synapse.
        Called by QueryScheduler; awaits the full response without streaming.
        """
        prompt = query.get("query", "")
        params = {k: v for k, v in query.items() if k != "query"}

        response, _, _ = await self.call_miner(prompt=prompt, params=params, uid=uid)
        return response

    async def x_search(
        self,
        query,
        uid: Optional[int] = None,
    ):
        """Receives question from user and returns the response from the miners."""

        try:
            prompt = query.get("query", "")
            params = {key: value for key, value in query.items() if key != "query"}

            response, selected_uid, axon = await self.call_miner(
                prompt=prompt, params=params, uid=uid
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
        uid: Optional[int] = None,
    ):
        """
        Perform a Twitter search using a specific tweet ID.
        """

        try:
            uid, axon = await self.neuron.get_random_miner(uid=uid)

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
        uid: Optional[int] = None,
    ):
        """
        Perform a Twitter search using multiple tweet URLs.
        """

        try:
            bt.logging.debug("run_task", "twitter urls search")

            uid, axon = await self.neuron.get_random_miner(uid=uid)

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
