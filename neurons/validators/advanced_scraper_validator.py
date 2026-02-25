import time
from typing import List, Optional, Tuple

import bittensor as bt
import torch

from desearch.dataset.date_filters import (
    DateFilter,
    DateFilterType,
    get_specified_date_filter,
)
from desearch.protocol import (
    ChatHistoryItem,
    Model,
    ResultType,
    ScraperStreamingSynapse,
)
from desearch.stream import collect_final_synapses
from desearch.utils import get_max_execution_time
from neurons.validators.base_validator import AbstractNeuron
from neurons.validators.penalty.chat_history_penalty import ChatHistoryPenaltyModel
from neurons.validators.penalty.exponential_penalty import ExponentialTimePenaltyModel
from neurons.validators.penalty.miner_score_penalty import MinerScorePenaltyModel
from neurons.validators.penalty.streaming_penalty import StreamingPenaltyModel
from neurons.validators.penalty.summary_rule_penalty import SummaryRulePenaltyModel
from neurons.validators.reward import RewardModelType, RewardScoringType
from neurons.validators.reward.performance_reward import PerformanceRewardModel
from neurons.validators.reward.reward_llm import RewardLLM
from neurons.validators.reward.search_content_relevance import (
    WebSearchContentRelevanceModel,
)
from neurons.validators.reward.summary_relevance import SummaryRelevanceRewardModel
from neurons.validators.reward.twitter_content_relevance import (
    TwitterContentRelevanceModel,
)
from neurons.validators.utils.mock import MockRewardModel


class AdvancedScraperValidator:
    def __init__(self, neuron: AbstractNeuron):
        self.neuron = neuron

        self.tools = [
            ["Twitter Search", "Reddit Search"],
            ["Twitter Search", "Web Search"],
            ["Twitter Search", "Web Search"],
            ["Twitter Search", "Web Search"],
            ["Twitter Search", "Web Search"],
            ["Twitter Search", "Hacker News Search"],
            ["Twitter Search", "Hacker News Search"],
            ["Twitter Search", "Youtube Search"],
            ["Twitter Search", "Youtube Search"],
            ["Twitter Search", "Youtube Search"],
            ["Twitter Search", "Web Search"],
            ["Twitter Search", "Reddit Search"],
            ["Twitter Search", "Reddit Search"],
            ["Twitter Search", "Hacker News Search"],
            ["Twitter Search", "ArXiv Search"],
            ["Twitter Search", "ArXiv Search"],
            ["Twitter Search", "Wikipedia Search"],
            ["Twitter Search", "Wikipedia Search"],
            ["Twitter Search", "Web Search"],
            ["Twitter Search", "Web Search"],
            ["Twitter Search", "Web Search"],
            ["Web Search"],
            ["Reddit Search"],
            ["Hacker News Search"],
            ["Youtube Search"],
            ["ArXiv Search"],
            ["Wikipedia Search"],
            ["Twitter Search", "Youtube Search", "ArXiv Search", "Wikipedia Search"],
            ["Twitter Search", "Web Search", "Reddit Search", "Hacker News Search"],
            [
                "Twitter Search",
                "Web Search",
                "Reddit Search",
                "Hacker News Search",
                "Youtube Search",
                "ArXiv Search",
                "Wikipedia Search",
            ],
        ]
        self.language = "en"
        self.region = "us"
        self.date_filter = "qdr:w"  # Past week

        # Init device.
        bt.logging.debug("loading", "device")
        bt.logging.debug(
            "self.neuron.config.neuron.device = ", str(self.neuron.config.neuron.device)
        )

        self.reward_weights = torch.tensor(
            [
                self.neuron.config.reward.twitter_content_weight,
                self.neuron.config.reward.web_search_relavance_weight,
                self.neuron.config.reward.summary_relevance_weight,
                self.neuron.config.reward.performance_weight,
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

        self.reward_llm = RewardLLM(self.neuron.config.neuron.scoring_model)

        self.reward_functions = [
            (
                TwitterContentRelevanceModel(
                    device=self.neuron.config.neuron.device,
                    scoring_type=RewardScoringType.summary_relevance_score_template,
                    llm_reward=self.reward_llm,
                    neuron=self.neuron,
                )
                if self.neuron.config.reward.twitter_content_weight > 0
                else MockRewardModel(RewardModelType.twitter_content_relevance.value)
            ),
            (
                WebSearchContentRelevanceModel(
                    device=self.neuron.config.neuron.device,
                    scoring_type=RewardScoringType.search_relevance_score_template,
                    llm_reward=self.reward_llm,
                    neuron=self.neuron,
                )
                if self.neuron.config.reward.web_search_relavance_weight > 0
                else MockRewardModel(RewardModelType.search_content_relevance.value)
            ),
            (
                SummaryRelevanceRewardModel(
                    device=self.neuron.config.neuron.device,
                    scoring_type=RewardScoringType.summary_relevance_score_template,
                    llm_reward=self.reward_llm,
                    neuron=self.neuron,
                )
                if self.neuron.config.reward.summary_relevance_weight > 0
                else MockRewardModel(RewardModelType.summary_relavance_match.value)
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
            StreamingPenaltyModel(max_penalty=1, neuron=self.neuron),
            ExponentialTimePenaltyModel(max_penalty=1, neuron=self.neuron),
            SummaryRulePenaltyModel(max_penalty=1, neuron=self.neuron),
            MinerScorePenaltyModel(max_penalty=1, neuron=self.neuron),
            ChatHistoryPenaltyModel(max_penalty=1, neuron=self.neuron),
        ]

    async def call_miner(
        self,
        prompt: str,
        date_filter: DateFilter,
        tools=[],
        language="en",
        region="us",
        google_date_filter="qdr:w",
        model: Optional[Model] = Model.NOVA,
        result_type: Optional[ResultType] = ResultType.LINKS_WITH_FINAL_SUMMARY,
        system_message: Optional[str] = None,
        scoring_system_message: Optional[str] = None,
        uid: Optional[int] = None,
        chat_history: Optional[List[ChatHistoryItem]] = [],
        count: Optional[int] = 10,
    ):
        max_execution_time = get_max_execution_time(model, count)

        start_time = time.time()

        uid, axon = await self.neuron.get_random_miner(uid=uid)
        uids = torch.tensor([uid])

        start_date = (
            date_filter.start_date.strftime("%Y-%m-%dT%H:%M:%SZ")
            if date_filter.start_date
            else None
        )
        end_date = (
            date_filter.end_date.strftime("%Y-%m-%dT%H:%M:%SZ")
            if date_filter.end_date
            else None
        )
        date_filter_type = date_filter.date_filter_type

        synapse = ScraperStreamingSynapse(
            prompt=prompt,
            model=model,
            start_date=start_date,
            end_date=end_date,
            date_filter_type=date_filter_type.value if date_filter_type else None,
            tools=tools,
            language=language,
            region=region,
            google_date_filter=google_date_filter,
            max_execution_time=max_execution_time,
            result_type=result_type,
            system_message=system_message,
            scoring_system_message=scoring_system_message,
            scoring_model=self.neuron.config.neuron.scoring_model,
            chat_history=chat_history,
            count=count,
        )

        timeout = max_execution_time + 5
        dendrite = next(self.neuron.dendrites)

        async_response = dendrite.call_stream(
            target_axon=axon,
            synapse=synapse.model_copy(),
            timeout=timeout,
            deserialize=False,
        )

        return async_response, uids, start_time

    async def compute_rewards_and_penalties(
        self,
        event,
        prompts: List[str],
        responses,
        uids,
        start_time,
        result_type: Optional[ResultType] = None,
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

            if result_type is None:
                result_type = ResultType.LINKS_WITH_FINAL_SUMMARY

            query_type = "synthetic"

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

            val_scores = []
            for val_score_responses, reward_function in zip(
                val_score_responses_list, self.reward_functions
            ):
                if reward_function.name in [
                    RewardModelType.twitter_content_relevance.value,
                    RewardModelType.search_content_relevance.value,
                ]:
                    val_scores.append(val_score_responses)

            for penalty_fn_i in self.penalty_functions:
                (
                    raw_penalty_i,
                    adjusted_penalty_i,
                    applied_penalty_i,
                ) = await penalty_fn_i.apply_penalties(
                    responses, uids, val_scores
                )
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
                "summary_reward": {},
                "twitter_reward": {},
                "search_reward": {},
                "latency_reward": {},
            }
            bt.logging.info(
                f"======================== Reward ==========================="
            )
            # Initialize an empty list to accumulate log messages
            log_messages = []
            for uid_tensor, reward, response in zip(uids, rewards.tolist(), responses):
                uid = uid_tensor.item()
                completion_length = (
                    len(response.completion) if response.completion is not None else 0
                )
                # Accumulate log messages instead of logging them immediately
                log_messages.append(
                    f"UID: {uid}, R: {round(reward, 3)}, C: {completion_length}"
                )
                bt.logging.trace(f"{response.completion}")

            # Log the accumulated messages in groups of three
            for i in range(0, len(log_messages), 3):
                bt.logging.info(" | ".join(log_messages[i : i + 3]))

            bt.logging.info(
                f"======================== Reward ==========================="
            )

            twitter_rewards = all_rewards[0]
            search_rewards = all_rewards[1]
            summary_rewards = all_rewards[2]
            latency_rewards = all_rewards[3]
            zipped_rewards = zip(
                uids,
                rewards.tolist(),
                responses,
                summary_rewards,
                twitter_rewards,
                search_rewards,
                latency_rewards,
            )

            for (
                uid_tensor,
                reward,
                response,
                summary_reward,
                twitter_reward,
                search_reward,
                latency_reward,
            ) in zipped_rewards:
                uid = uid_tensor.item()  # Convert tensor to int
                uid_scores_dict[uid] = reward
                scores[uid] = reward  # Now 'uid' is an int, which is a valid key type
                wandb_data["scores"][uid] = reward
                wandb_data["responses"][uid] = response.completion
                wandb_data["prompts"][uid] = response.prompt
                wandb_data["summary_reward"][uid] = summary_reward
                wandb_data["twitter_reward"][uid] = twitter_reward
                wandb_data["search_reward"][uid] = search_reward
                wandb_data["latency_reward"][uid] = latency_reward

            await self.neuron.update_scores(
                wandb_data=wandb_data,
                responses=responses,
                uids=uids,
                rewards=rewards,
                all_rewards=all_rewards,
                all_original_rewards=all_original_rewards,
                val_score_responses_list=val_score_responses_list,
                neuron=self.neuron,
                query_type=query_type,
            )

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
    ) -> Tuple[Optional[object], dict]:
        """
        Send a scoring query to a specific miner and return (response, task).
        Called by QueryScheduler; collects the full synapse.
        """
        prompt = query["query"]
        tools = query.get("tools", [])
        date_filter = get_specified_date_filter(
            DateFilterType(query.get("date_filter_type", DateFilterType.PAST_WEEK.value))
        )

        async_response, uids, start_time = await self.call_miner(
            prompt=prompt,
            date_filter=date_filter,
            tools=tools,
            language=self.language,
            region=self.region,
            google_date_filter=self.date_filter,
            model=Model.NOVA,
            uid=uid,
        )

        final_synapses = await collect_final_synapses([async_response], uids, start_time)
        response = final_synapses[0] if final_synapses else None
        return response, prompt

    async def organic(
        self,
        query,
        model: Optional[Model] = Model.NOVA,
        uid: Optional[int] = None,
        result_type: Optional[ResultType] = ResultType.LINKS_WITH_FINAL_SUMMARY,
        is_collect_final_synapses: bool = False,  # Flag to collect final synapses
    ):
        """Receives question from user and returns the response from the miners."""

        try:
            prompt = query["content"]
            tools = query.get("tools", [])
            date_filter = query.get("date_filter", DateFilterType.PAST_WEEK.value)
            count = query.get("count")
            system_message = query.get("system_message")
            scoring_system_message = query.get("scoring_system_message")
            chat_history = query.get("chat_history", [])
            start_date = query.get("start_date")
            end_date = query.get("end_date")

            if start_date or end_date:
                date_filter = DateFilter(start_date=start_date, end_date=end_date)
            elif isinstance(date_filter, str):
                date_filter_type = DateFilterType(date_filter)
                date_filter = get_specified_date_filter(date_filter_type)

            async_response, uids, start_time = await self.call_miner(
                prompt=prompt,
                tools=tools,
                language=self.language,
                region=self.region,
                date_filter=date_filter,
                google_date_filter=self.date_filter,
                model=model,
                result_type=result_type,
                system_message=system_message,
                scoring_system_message=scoring_system_message,
                uid=uid,
                chat_history=chat_history,
                count=count,
            )

            final_synapses = []

            if is_collect_final_synapses:
                # Collect specified uids from responses and score
                final_synapses = await collect_final_synapses(
                    [async_response], uids, start_time
                )

                for synapse in final_synapses:
                    yield synapse
            else:
                # Stream miner response to the UI
                async for value in async_response:
                    if isinstance(value, bt.Synapse):
                        final_synapses.append(value)
                    else:
                        yield value
        except Exception as e:
            bt.logging.error(f"Error in organic: {e}")
            raise e
