import random
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import bittensor as bt
import pytz
import torch

from desearch.protocol import (
    TwitterIDSearchSynapse,
    TwitterSearchSynapse,
    TwitterURLsSearchSynapse,
)
from desearch.synapse import collect_responses
from neurons.validators.base_validator import AbstractNeuron
from neurons.validators.penalty.exponential_penalty import ExponentialTimePenaltyModel
from neurons.validators.penalty.twitter_count_penalty import TwitterCountPenaltyModel
from neurons.validators.reward import RewardModelType, RewardScoringType
from neurons.validators.reward.performance_reward import PerformanceRewardModel
from neurons.validators.reward.twitter_basic_search_content_relevance import (
    TwitterBasicSearchContentRelevanceModel,
)
from neurons.validators.utils.mock import MockRewardModel
from neurons.validators.utils.tasks import SearchTask


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

        # Hardcoded weights here because the advanced scraper validator implementation is based on args.
        self.twitter_content_weight = 0.70
        self.performance_weight = 0.30

        self.reward_weights = torch.tensor(
            [
                self.twitter_content_weight,
                self.performance_weight,
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
            (
                TwitterBasicSearchContentRelevanceModel(
                    device=self.neuron.config.neuron.device,
                    scoring_type=RewardScoringType.search_relevance_score_template,
                    neuron=self.neuron,
                )
                if self.neuron.config.reward.twitter_content_weight > 0
                else MockRewardModel(RewardModelType.twitter_content_relevance.value)
            ),
            (
                PerformanceRewardModel(
                    device=self.neuron.config.neuron.device,
                    neuron=self.neuron,
                )
                if self.neuron.config.reward.performance_weight > 0
                else MockRewardModel(RewardModelType.performance_score.value)
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

    async def run_twitter_basic_search_and_score(
        self,
        tasks: List[SearchTask],
        params_list: List[Dict[str, Any]],
        uid: Optional[int] = None,
    ):
        event = {
            "names": [task.task_name for task in tasks],
            "task_types": [task.task_type for task in tasks],
        }

        start_time = time.time()

        uid, axon = await self.neuron.get_random_miner(uid=uid)
        uids = torch.tensor([uid])
        axons = [axon]

        synapses: List[TwitterSearchSynapse] = [
            TwitterSearchSynapse(
                **params,
                query=task.compose_prompt(),
                max_execution_time=self.calc_max_execution_time(params.get("count")),
            )
            for task, params in zip(tasks, params_list)
        ]

        all_tasks = []  # List to collect all asyncio tasks

        for axon, synapse in zip(axons, synapses):
            dendrite = next(self.neuron.dendrites)

            task = dendrite.call(
                target_axon=axon,
                synapse=synapse.model_copy(),
                timeout=synapse.max_execution_time + 5,
                deserialize=False,
            )
            all_tasks.append(task)

        return all_tasks, uids, event, start_time

    async def compute_rewards_and_penalties(
        self,
        event,
        tasks,
        responses,
        uids,
        start_time,
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
                ) = await penalty_fn_i.apply_penalties(responses, tasks, uids)
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
            self.log_event(tasks, event, start_time, uids, rewards)

            scores = torch.zeros(len(self.neuron.metagraph.hotkeys))
            uid_scores_dict = {}
            wandb_data = {
                "modality": "twitter_scrapper",
                "prompts": {},
                "responses": {},
                "scores": {},
                "timestamps": {},
                "twitter_reward": {},
                "latency_reward": {},
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
            latency_rewards = all_rewards[1]
            zipped_rewards = zip(
                uids,
                rewards.tolist(),
                responses,
                twitter_rewards,
                latency_rewards,
            )

            for (
                uid_tensor,
                reward,
                response,
                twitter_reward,
                latency_reward,
            ) in zipped_rewards:
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
                wandb_data["latency_reward"][uid] = latency_reward

            await self.neuron.update_scores_for_basic(
                wandb_data=wandb_data,
                responses=responses,
                uids=uids,
                rewards=rewards,
                all_rewards=all_rewards,
                all_original_rewards=all_original_rewards,
                val_score_responses_list=val_score_responses_list,
                neuron=self.neuron,
            )

            return rewards, uids, val_score_responses_list, event, all_original_rewards
        except Exception as e:
            bt.logging.error(f"Error in compute_rewards_and_penalties: {e}")
            raise e

    def log_event(self, tasks, event, start_time, uids, rewards):
        event.update(
            {
                "step_length": time.time() - start_time,
                "prompts": [task.compose_prompt() for task in tasks],
                "uids": uids.tolist(),
                "rewards": rewards.tolist(),
            }
        )

        bt.logging.debug("Run Task event:", event)

    def generate_random_twitter_search_params(self) -> Dict[str, Any]:
        """
        Generate random logical parameters for Twitter search queries.
        Returns a dictionary with randomly selected parameters.
        """

        # Define which fields will be used (randomly select 1-6 fields)
        all_fields = [
            "is_quote",
            "is_video",
            "is_image",
            "min_retweets",
            "min_replies",
            "min_likes",
            "date_range",
        ]

        num_fields = 1
        selected_fields = random.sample(all_fields, num_fields)

        params: Dict[str, Any] = {}

        THREE_YEAR_IN_DAYS = 3 * 365

        # Generate random date range if selected
        if "date_range" in selected_fields:
            # Generate end date in past three years
            end_date = datetime.now(pytz.UTC) - timedelta(
                days=random.randint(0, THREE_YEAR_IN_DAYS)
            )

            # Randomly choose time window
            start_date = end_date - timedelta(days=random.randint(7, 14))

            params["start_date"] = start_date.strftime("%Y-%m-%d_%H:%M:%S_UTC")
            params["end_date"] = end_date.strftime("%Y-%m-%d_%H:%M:%S_UTC")

        # Handle media type flags (ensuring is_video and is_image aren't both True)
        if "is_video" in selected_fields and "is_image" in selected_fields:
            # If both selected, ensure they're not both True
            video_val = random.choice([True, False])

            params["is_video"] = video_val

            if video_val is False:
                params["is_image"] = random.choice([True, False])
        elif "is_video" in selected_fields:
            params["is_video"] = random.choice([True, False])
        elif "is_image" in selected_fields:
            params["is_image"] = random.choice([True, False])

        # Handle quote status
        if "is_quote" in selected_fields:
            params["is_quote"] = random.choice([True, False])

        # Handle engagement metrics with logical ranges
        if "min_likes" in selected_fields:
            params["min_likes"] = random.randint(5, 100)
        if "min_replies" in selected_fields:
            params["min_replies"] = random.randint(5, 20)
        if "min_retweets" in selected_fields:
            params["min_retweets"] = random.randint(5, 20)

        return params

    async def send_scoring_query(
        self,
        query: dict,
        uid: int,
    ) -> Tuple[Optional[object], SearchTask]:
        """
        Send a scoring query to a specific miner and return (response, task).
        Called by QueryScheduler; awaits the full response without streaming.
        """
        prompt = query.get("query", "")
        params = {k: v for k, v in query.items() if k != "query"}

        task = SearchTask(
            base_text=prompt,
            task_name="twitter search",
            task_type="twitter_search",
            criteria=[],
        )

        all_tasks, uids, event, start_time = await self.run_twitter_basic_search_and_score(
            tasks=[task],
            params_list=[params],
            uid=uid,
        )

        responses = await collect_responses(all_tasks)
        response = responses[0] if responses else None
        return response, task

    async def organic(
        self,
        query,
        uid: Optional[int] = None,
    ):
        """Receives question from user and returns the response from the miners."""

        try:
            prompt = query.get("query", "")

            tasks = [
                SearchTask(
                    base_text=prompt,
                    task_name="twitter search",
                    task_type="twitter_search",
                    criteria=[],
                )
            ]

            (
                async_responses,
                uids,
                event,
                start_time,
            ) = await self.run_twitter_basic_search_and_score(
                tasks=tasks,
                params_list=[
                    {key: value for key, value in query.items() if key != "query"}
                ],
                uid=uid,
            )

            final_responses = []

            # Process responses and collect successful ones
            for async_response in async_responses:
                response = await async_response
                if response:
                    final_responses.append(response)
                    yield response
                else:
                    bt.logging.warning(
                        f"Invalid response for UID: {response.axon.hotkey if response else 'Unknown'}"
                    )

        except Exception as e:
            bt.logging.error(f"Error in organic: {e}")
            raise e

    async def twitter_id_search(
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

            return synapse.results
        except Exception as e:
            bt.logging.error(f"Error in ID search: {e}")
            raise e

    async def twitter_urls_search(
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

            return synapse.results
        except Exception as e:
            bt.logging.error(f"Error in URLs search: {e}")
            raise e
