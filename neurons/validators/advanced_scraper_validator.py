import torch
import random
import asyncio
import time
from typing import List, Optional
import bittensor as bt
from datura.stream import collect_final_synapses
from reward import RewardModelType, RewardScoringType
from utils.mock import MockRewardModel

from datura.dataset import QuestionsDataset
from datura.dataset.tool_return import ResponseOrder
from datura.dataset.date_filters import (
    get_random_date_filter,
    get_specified_date_filter,
    DateFilterType,
)
from datura import QUERY_MINERS
from datura.protocol import Model, ResultType, ScraperStreamingSynapse
from datura.utils import get_max_execution_time
from neurons.validators.base_validator import AbstractNeuron
from neurons.validators.reward.summary_relevance import SummaryRelevanceRewardModel
from neurons.validators.reward.twitter_content_relevance import (
    TwitterContentRelevanceModel,
)
from neurons.validators.reward.search_content_relevance import (
    WebSearchContentRelevanceModel,
)
from neurons.validators.reward.performance_reward import PerformanceRewardModel
from neurons.validators.reward.reward_llm import RewardLLM
from neurons.validators.utils.tasks import TwitterTask
from neurons.validators.organic_query_state import OrganicQueryState
from neurons.validators.penalty.streaming_penalty import StreamingPenaltyModel
from neurons.validators.penalty.exponential_penalty import ExponentialTimePenaltyModel


class AdvancedScraperValidator:
    def __init__(self, neuron: AbstractNeuron):
        self.neuron = neuron
        self.timeout = 180
        self.execution_time_options = [Model.NOVA, Model.ORBIT, Model.HORIZON]
        self.execution_time_probabilities = [0.8, 0.1, 0.1]

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

        self.synthetic_history = []
        self.organic_query_state = OrganicQueryState()
        self.organic_history = []  # New list to track organic queries
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

        self.reward_llm = RewardLLM()

        self.reward_functions = [
            (
                TwitterContentRelevanceModel(
                    device=self.neuron.config.neuron.device,
                    scoring_type=RewardScoringType.summary_relevance_score_template,
                    llm_reward=self.reward_llm,
                )
                if self.neuron.config.reward.twitter_content_weight > 0
                else MockRewardModel(RewardModelType.twitter_content_relevance.value)
            ),
            (
                WebSearchContentRelevanceModel(
                    device=self.neuron.config.neuron.device,
                    scoring_type=RewardScoringType.search_relevance_score_template,
                    llm_reward=self.reward_llm,
                )
                if self.neuron.config.reward.web_search_relavance_weight > 0
                else MockRewardModel(RewardModelType.search_content_relevance.value)
            ),
            (
                SummaryRelevanceRewardModel(
                    device=self.neuron.config.neuron.device,
                    scoring_type=RewardScoringType.summary_relevance_score_template,
                    llm_reward=self.reward_llm,
                )
                if self.neuron.config.reward.summary_relevance_weight > 0
                else MockRewardModel(RewardModelType.summary_relavance_match.value)
            ),
            (
                PerformanceRewardModel(
                    device=self.neuron.config.neuron.device,
                )
                if self.neuron.config.reward.performance_weight > 0
                else MockRewardModel(RewardModelType.performance_score.value)
            ),
        ]

        self.penalty_functions = [
            StreamingPenaltyModel(max_penalty=1),
            ExponentialTimePenaltyModel(max_penalty=1),
        ]

    def get_random_execution_time(self):
        return random.choices(
            self.execution_time_options, self.execution_time_probabilities
        )[0]

    async def run_task_and_score(
        self,
        tasks: List[TwitterTask],
        strategy=QUERY_MINERS.RANDOM,
        is_only_allowed_miner=True,
        specified_uids=None,
        date_filter=None,
        tools=[],
        language="en",
        region="us",
        google_date_filter="qdr:w",
        response_order=ResponseOrder.SUMMARY_FIRST,
        model: Optional[Model] = Model.NOVA,
        result_type: Optional[ResultType] = ResultType.LINKS_WITH_SUMMARIES,
    ):
        max_execution_time = get_max_execution_time(model)

        # Record event start time.
        event = {
            "names": [task.task_name for task in tasks],
            "task_types": [task.task_type for task in tasks],
        }
        start_time = time.time()

        # Get random id on that step
        uids = await self.neuron.get_uids(
            strategy=strategy,
            is_only_allowed_miner=is_only_allowed_miner,
            specified_uids=specified_uids,
        )

        start_date = date_filter.start_date.strftime("%Y-%m-%dT%H:%M:%SZ")
        end_date = date_filter.end_date.strftime("%Y-%m-%dT%H:%M:%SZ")
        axons = [self.neuron.metagraph.axons[uid] for uid in uids]

        synapses = [
            ScraperStreamingSynapse(
                prompt=task.compose_prompt(),
                model=model,
                start_date=start_date,
                end_date=end_date,
                date_filter_type=date_filter.date_filter_type.value,
                tools=tools,
                language=language,
                region=region,
                google_date_filter=google_date_filter,
                response_order=response_order.value,
                max_execution_time=max_execution_time,
                result_type=result_type,
            )
            for task in tasks
        ]

        axon_groups = [axons[:80], axons[80:160], axons[160:]]
        synapse_groups = [synapses[:80], synapses[80:160], synapses[160:]]
        dendrites = [
            self.neuron.dendrite1,
            self.neuron.dendrite2,
            self.neuron.dendrite3,
        ]

        async_responses = []
        timeout = max_execution_time + 5

        for dendrite, axon_group, synapse_group in zip(
            dendrites, axon_groups, synapse_groups
        ):
            async_responses.extend(
                [
                    dendrite.call_stream(
                        target_axon=axon,
                        synapse=synapse.copy(),
                        timeout=timeout,
                        deserialize=False,
                    )
                    for axon, synapse in zip(axon_group, synapse_group)
                ]
            )

        return async_responses, uids, event, start_time

    async def compute_rewards_and_penalties(
        self,
        event,
        tasks,
        responses,
        uids,
        start_time,
        is_synthetic=False,
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

            organic_penalties = []

            if result_type is None:
                result_type = ResultType.LINKS_WITH_SUMMARIES

            if is_synthetic:
                penalized_uids = []

                for uid, response in zip(uids.tolist(), responses):
                    has_penalty = self.organic_query_state.has_penalty(
                        response.axon.hotkey
                    )

                    organic_penalties.append(has_penalty)

                    if has_penalty:
                        penalized_uids.append(uid)

                bt.logging.info(
                    f"Following UIDs will be penalized as they failed organic query: {penalized_uids}"
                )
            else:
                organic_penalties = [False] * len(uids)

            query_type = "synthetic" if is_synthetic else "organic"

            for weight_i, reward_fn_i in zip(
                self.reward_weights, self.reward_functions
            ):
                start_time = time.time()
                (
                    reward_i_normalized,
                    reward_event,
                    val_score_responses,
                    original_rewards,
                ) = await reward_fn_i.apply(responses, uids, organic_penalties)

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
                raw_penalty_i, adjusted_penalty_i, applied_penalty_i = (
                    penalty_fn_i.apply_penalties(responses, tasks)
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

            if is_synthetic:
                scattered_rewards = self.neuron.update_moving_averaged_scores(
                    uids, rewards
                )
                self.log_event(tasks, event, start_time, uids, rewards)

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
                completion_links_length = (
                    len(response.completion_links)
                    if response.completion_links is not None
                    else 0
                )
                # Accumulate log messages instead of logging them immediately
                log_messages.append(
                    f"UID: {uid}, R: {round(reward, 3)}, C: {completion_length}, L: {completion_links_length}"
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
                organic_penalties=organic_penalties,
                neuron=self.neuron,
                query_type=query_type,
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

    async def query_and_score(self, strategy=QUERY_MINERS.RANDOM):
        try:
            if not len(self.neuron.available_uids):
                bt.logging.info("No available UIDs, skipping task execution.")
                return

            # Cleanup organic history older than 4 hours
            current_time = time.time()
            four_hours_ago = current_time - 4 * 3600
            self.organic_history = [entry for entry in self.organic_history if entry['timestamp'] >= four_hours_ago]

            # Generate synthetic parameters
            tools = random.choice(self.tools)
            random_model = self.get_random_execution_time()
            date_filter = get_random_date_filter()
            synthetic_date_filter_type = date_filter.date_filter_type

            # Determine UIDs to exclude (matching model, tools, date_filter)
            matching_uids = set()
            for entry in self.organic_history:
                if (entry['model'] == random_model and
                    entry['tools'] == tools and
                    entry['date_filter_type'] == synthetic_date_filter_type.value):
                    matching_uids.add(entry['uid'])

            available_uids = self.neuron.available_uids.copy()
            synthetic_uids = [uid for uid in available_uids if uid not in matching_uids]

            dataset = QuestionsDataset()
            prompts = await asyncio.gather(
                *[
                    dataset.generate_new_question_with_openai(tools)
                    for _ in range(len(synthetic_uids))
                ]
            )

            tasks = [
                TwitterTask(
                    base_text=prompt,
                    task_name="augment",
                    task_type="twitter_scraper",
                    criteria=[],
                )
                for prompt in prompts
            ]

            bt.logging.debug(
                f"Query and score running with prompts: {prompts} and tools: {tools}"
            )

            max_execution_time = get_max_execution_time(random_model)

            async_responses, uids, event, start_time = await self.run_task_and_score(
                tasks=tasks,
                strategy=strategy,
                is_only_allowed_miner=False,
                date_filter=date_filter,
                tools=tools,
                language=self.language,
                region=self.region,
                google_date_filter=self.date_filter,
                model=random_model,
            )

            final_synapses = await collect_final_synapses(
                async_responses, uids, start_time, max_execution_time
            )

            # Collect organic responses for matching UIDs
            organic_responses = {}
            for entry in self.organic_history:
                if (entry['uid'] in matching_uids and
                    entry['model'] == random_model and
                    entry['tools'] == tools and
                    entry['date_filter_type'] == synthetic_date_filter_type.value):
                    organic_responses[entry['uid']] = entry['response']

            # Merge synthetic and organic responses in UID order
            merged_responses = []
            merged_uids = []
            synthetic_responses_dict = {uid.item(): response for uid, response in zip(uids, final_synapses)}

            for uid in available_uids:
                if uid in organic_responses:
                    merged_responses.append(organic_responses[uid])
                    merged_uids.append(uid)
                elif uid in synthetic_responses_dict:
                    merged_responses.append(synthetic_responses_dict[uid])
                    merged_uids.append(uid)

            # Remove used organic entries
            self.organic_history = [
                entry for entry in self.organic_history
                if not (entry['uid'] in matching_uids and
                        entry['model'] == random_model and
                        entry['tools'] == tools and
                        entry['date_filter_type'] == synthetic_date_filter_type.value)
            ]

            # Update uids and responses
            uids = torch.tensor(merged_uids, device=self.neuron.config.neuron.device)
            final_synapses = merged_responses

            self.synthetic_history.append(
                (event, tasks, final_synapses, uids, start_time)
            )

            await self.score_random_synthetic_query()

        except Exception as e:
            bt.logging.error(f"Error in query_and_score: {e}")
            raise e

    async def score_random_synthetic_query(self):
        # Collect synthetic queries and score randomly
        synthetic_queries_collection_size = 2

        if len(self.synthetic_history) < synthetic_queries_collection_size:
            bt.logging.info(
                f"Skipping scoring random synthetic query as history length is {len(self.synthetic_history)}"
            )

            return

        event, tasks, final_synapses, uids, start_time = random.choice(
            self.synthetic_history
        )

        bt.logging.info(f"Scoring random synthetic query: {event}")

        await self.compute_rewards_and_penalties(
            event=event,
            tasks=tasks,
            responses=final_synapses,
            uids=uids,
            start_time=start_time,
            is_synthetic=True,
        )

        self.synthetic_history = []

    async def organic(
        self,
        query,
        model: Optional[Model] = Model.NOVA,
        random_synapse: ScraperStreamingSynapse = None,
        random_uid=None,
        specified_uids=None,
        result_type: Optional[ResultType] = ResultType.LINKS_WITH_SUMMARIES,
        is_collect_final_synapses: bool = False,  # Flag to collect final synapses
    ):
        """Receives question from user and returns the response from the miners."""
        max_execution_time = get_max_execution_time(model)

        if not len(self.neuron.available_uids):
            bt.logging.info("Not available uids")
            raise StopAsyncIteration("Not available uids")

        is_interval_query = random_synapse is not None

        try:
            prompt = query["content"]
            tools = query.get("tools", [])
            date_filter = query.get("date_filter", DateFilterType.PAST_WEEK.value)
            response_order = query.get("response_order", ResponseOrder.LINKS_FIRST)

            if isinstance(date_filter, str):
                date_filter_type = DateFilterType(date_filter)
                date_filter = get_specified_date_filter(date_filter_type)

            tasks = [
                TwitterTask(
                    base_text=prompt,
                    task_name="augment",
                    task_type="twitter_scraper",
                    criteria=[],
                )
            ]

            async_responses, uids, event, start_time = await self.run_task_and_score(
                tasks=tasks,
                strategy=QUERY_MINERS.ALL if specified_uids else QUERY_MINERS.RANDOM,
                is_only_allowed_miner=self.neuron.config.subtensor.network != "finney",
                tools=tools,
                language=self.language,
                region=self.region,
                date_filter=date_filter,
                google_date_filter=self.date_filter,
                specified_uids=specified_uids,
                response_order=response_order,
                model=model,
                result_type=result_type,
            )

            final_synapses = []

            if specified_uids or is_collect_final_synapses:
                # Collect specified uids from responses and score
                final_synapses = await collect_final_synapses(
                    async_responses, uids, start_time, max_execution_time
                )

                if is_collect_final_synapses:
                    for synapse in final_synapses:
                        yield synapse
            else:
                # Stream random miner to the UI
                for response in async_responses:
                    async for value in response:
                        if isinstance(value, bt.Synapse):
                            final_synapses.append(value)
                        else:
                            yield value

            async def process_and_score_responses(uids):
                if is_interval_query:
                    # Add the random_synapse to final_synapses and its UID to uids
                    final_synapses.append(random_synapse)
                    uids = torch.cat([uids, torch.tensor([random_uid])])

                _, _, _, _, original_rewards = await self.compute_rewards_and_penalties(
                    event=event,
                    tasks=tasks,
                    responses=final_synapses,
                    uids=uids,
                    start_time=start_time,
                    is_synthetic=False,
                )

                if not is_interval_query:
                    self.organic_query_state.save_organic_queries(
                        final_synapses, uids, original_rewards
                    )
                # After scoring, save to organic_history
                current_time = time.time()
                for uid_tensor, response in zip(uids, final_synapses):
                    self.organic_history.append({
                        'uid': uid_tensor.item(),
                        'model': response.model,
                        'tools': response.tools,
                        'date_filter_type': response.date_filter_type,
                        'timestamp': current_time,
                        'response': response
                    })

            asyncio.create_task(process_and_score_responses(uids))
        except Exception as e:
            bt.logging.error(f"Error in organic: {e}")
            raise e