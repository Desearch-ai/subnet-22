import torch
import asyncio
import time
from typing import Any, Dict, List
import bittensor as bt
from datura.protocol import (
    PeopleSearchSynapse,
)
from datura.synapse import collect_responses
from neurons.validators.utils.mock import MockRewardModel
from datura.dataset import QuestionsDataset
from datura import QUERY_MINERS
from neurons.validators.base_validator import AbstractNeuron
from neurons.validators.reward import RewardModelType, RewardScoringType
from neurons.validators.reward.performance_reward import PerformanceRewardModel
from neurons.validators.reward.people_search_relevance import PeopleSearchRelevanceModel
from neurons.validators.utils.tasks import SearchTask
from neurons.validators.basic_organic_query_state import BasicOrganicQueryState
from neurons.validators.penalty.exponential_penalty import ExponentialTimePenaltyModel
from neurons.validators.organic_history_mixin import OrganicHistoryMixin
from neurons.validators.utils.prompt.search_criteria_generate_prompt import (
    SearchCriteriaGeneratePrompt,
)
from neurons.validators.utils.prompt.people_search_question_generate_prompt import (
    PeopleSearchQuestionGeneratePrompt,
)


class PeopleSearchValidator(OrganicHistoryMixin):
    def __init__(self, neuron: AbstractNeuron):
        super().__init__()

        self.neuron = neuron
        self.timeout = 180
        self.max_execution_time = 100

        self.basic_organic_query_state = BasicOrganicQueryState()

        # Init device.
        bt.logging.debug("loading", "device")
        bt.logging.debug(
            "self.neuron.config.neuron.device = ", str(self.neuron.config.neuron.device)
        )

        # Hardcoded weights here because the advanced scraper validator implementation is based on args.
        self.people_search_weight = 0.70
        self.performance_weight = 0.30

        self.reward_weights = torch.tensor(
            [
                self.people_search_weight,
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
                PeopleSearchRelevanceModel(
                    device=self.neuron.config.neuron.device,
                    scoring_type=RewardScoringType.search_relevance_score_template,
                )
                if self.neuron.config.reward.people_search_relavance_weight > 0
                else MockRewardModel(RewardModelType.people_search_relevance.value)
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
            ExponentialTimePenaltyModel(max_penalty=1),
        ]

    async def generate_criteria(self, synapse: PeopleSearchSynapse):
        search_criteria_prompt = SearchCriteriaGeneratePrompt()
        response = await search_criteria_prompt.get_response(synapse.query)

        synapse.criteria = [text.strip() for text in response.splitlines()]

    async def run_people_search_and_score(
        self,
        tasks: List[SearchTask],
        params_list: List[Dict[str, Any]],
        strategy=QUERY_MINERS.RANDOM,
        is_only_allowed_miner=True,
        specified_uids=None,
        is_synthetic=False,
    ):
        event = {
            "names": [task.task_name for task in tasks],
            "task_types": [task.task_type for task in tasks],
        }

        start_time = time.time()

        if is_synthetic:
            uids = await self.neuron.get_uids(
                strategy=strategy,
                is_only_allowed_miner=is_only_allowed_miner,
                specified_uids=specified_uids,
            )
            axons = [self.neuron.metagraph.axons[uid] for uid in uids]
        else:
            uid, axon = await self.neuron.get_random_miner()
            uids = torch.tensor([uid])
            axons = [axon]

        synapses: List[PeopleSearchSynapse] = [
            PeopleSearchSynapse(
                **params,
                query=task.compose_prompt(),
                max_execution_time=self.max_execution_time,
                is_synthetic=is_synthetic,
            )
            for task, params in zip(tasks, params_list)
        ]

        await collect_responses(
            [self.generate_criteria(synapse) for synapse in synapses]
        )

        dendrites = [
            self.neuron.dendrite1,
            self.neuron.dendrite2,
            self.neuron.dendrite3,
        ]

        axon_groups = [axons[:80], axons[80:160], axons[160:]]
        synapse_groups = [synapses[:80], synapses[80:160], synapses[160:]]

        all_tasks = []  # List to collect all asyncio tasks
        timeout = self.max_execution_time + 5

        for dendrite, axon_group, synapse_group in zip(
            dendrites, axon_groups, synapse_groups
        ):
            for axon, syn in zip(axon_group, synapse_group):
                # Create a task for each dendrite call
                task = dendrite.call(
                    target_axon=axon,
                    synapse=syn.model_copy(),
                    timeout=timeout,
                    deserialize=False,
                )
                all_tasks.append(task)

        # Await all tasks concurrently
        all_responses = await asyncio.gather(*all_tasks, return_exceptions=True)

        return all_responses, uids, event, start_time

    async def compute_rewards_and_penalties(
        self,
        event,
        tasks,
        responses,
        uids,
        start_time,
        is_synthetic=False,
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

            bt.logging.trace(f"Received responses: {responses}")

            if is_synthetic:
                penalized_uids = []

                for uid, response in zip(uids.tolist(), responses):
                    has_penalty = await self.basic_organic_query_state.has_penalty(
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
                    await penalty_fn_i.apply_penalties(responses, tasks)
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
                scattered_rewards = await self.neuron.update_moving_averaged_scores(
                    uids, rewards
                )
                self.log_event(tasks, event, start_time, uids, rewards)

            scores = torch.zeros(len(self.neuron.metagraph.hotkeys))
            uid_scores_dict = {}
            wandb_data = {
                "modality": "people_scrapper",
                "prompts": {},
                "responses": {},
                "scores": {},
                "timestamps": {},
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

                # Accumulate log messages instead of logging them immediately
                log_messages.append(f"UID: {uid}, R: {round(reward, 3)}")

            # Log the accumulated messages in groups of three
            for i in range(0, len(log_messages), 3):
                bt.logging.info(" | ".join(log_messages[i : i + 3]))

            bt.logging.info(
                f"======================== Reward ==========================="
            )
            bt.logging.info(f"this is a all reward {all_rewards} ")

            search_rewards = all_rewards[0]
            latency_rewards = all_rewards[1]
            zipped_rewards = zip(
                uids,
                rewards.tolist(),
                responses,
                search_rewards,
                latency_rewards,
            )

            for (
                uid_tensor,
                reward,
                response,
                search_reward,
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
                wandb_data["search_reward"][uid] = search_reward
                wandb_data["latency_reward"][uid] = latency_reward

            await self.neuron.update_scores_for_basic(
                wandb_data=wandb_data,
                responses=responses,
                uids=uids,
                rewards=rewards,
                all_rewards=all_rewards,
                all_original_rewards=all_original_rewards,
                val_score_responses_list=val_score_responses_list,
                organic_penalties=organic_penalties,
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

    async def generate_prompt_with_openai(self):
        return await PeopleSearchQuestionGeneratePrompt().get_response()

    async def query_and_score(self, strategy, specified_uids=None):
        try:
            # Question generation
            prompts = await asyncio.gather(
                *[
                    self.generate_prompt_with_openai()
                    for _ in range(
                        len(
                            specified_uids
                            if specified_uids
                            else self.neuron.metagraph.uids
                        )
                    )
                ]
            )

            params = [{} for _ in range(len(prompts))]

            # 2) Build tasks from the generated prompts
            tasks = [
                SearchTask(
                    base_text=prompt,
                    task_name="people search",
                    task_type="people_search",
                    criteria=[],
                )
                for prompt in prompts
            ]

            bt.logging.debug(
                f"[query_and_score_people_search] Running with prompts: {prompts}"
            )

            # 4) Run the basic people search
            responses, uids, event, start_time = await self.run_people_search_and_score(
                tasks=tasks,
                strategy=strategy,
                is_only_allowed_miner=False,
                specified_uids=specified_uids,
                params_list=params,
                is_synthetic=True,
            )

            if self.neuron.config.neuron.synthetic_disabled:
                await self._save_organic_response(
                    uids, responses, tasks, event, start_time
                )
            else:
                await self.compute_rewards_and_penalties(
                    event=event,
                    tasks=tasks,
                    responses=responses,
                    uids=uids,
                    start_time=start_time,
                    is_synthetic=True,
                )
        except Exception as e:
            bt.logging.error(f"Error in query_and_score_people_search: {e}")
            raise

    async def organic(
        self,
        query,
        random_synapse: PeopleSearchSynapse = None,
        random_uid=None,
        specified_uids=None,
    ):
        """Receives question from user and returns the response from the miners."""

        is_interval_query = random_synapse is not None

        try:
            prompt = query.get("query", "")

            tasks = [
                SearchTask(
                    base_text=prompt,
                    task_name="people search",
                    task_type="people_search",
                    criteria=[],
                )
            ]

            async_responses, uids, event, start_time = (
                await self.run_people_search_and_score(
                    tasks=tasks,
                    strategy=(
                        QUERY_MINERS.ALL if specified_uids else QUERY_MINERS.RANDOM
                    ),
                    is_only_allowed_miner=self.neuron.config.subtensor.network
                    != "finney",
                    specified_uids=specified_uids,
                    params_list=[
                        {key: value for key, value in query.items() if key != "query"}
                    ],
                )
            )

            final_responses = []

            # Process responses and collect successful ones
            for response in async_responses:
                if response:
                    final_responses.append(response)
                    yield response
                else:
                    bt.logging.warning(
                        f"Invalid response for UID: {response.axon.hotkey if response else 'Unknown'}"
                    )

            async def process_and_score_responses(uids):
                if is_interval_query:
                    # Add the random_synapse to final_responses and its UID to uids
                    final_responses.append(random_synapse)
                    uids = torch.cat([uids, torch.tensor([random_uid])])

                # Compute rewards and penalties
                if not self.neuron.config.neuron.synthetic_disabled:
                    _, _, _, _, original_rewards = (
                        await self.compute_rewards_and_penalties(
                            event=event,
                            tasks=tasks,
                            responses=final_responses,
                            uids=uids,
                            start_time=start_time,
                            is_synthetic=False,
                        )
                    )

                    # Save organic queries if not an interval query
                    if not is_interval_query:
                        await self.basic_organic_query_state.save_organic_queries(
                            final_responses, uids, original_rewards
                        )

                if (
                    self.neuron.config.neuron.synthetic_disabled
                    and not is_interval_query
                ):
                    await self._save_organic_response(
                        uids, final_responses, tasks, event, start_time
                    )

            # Schedule scoring task
            asyncio.create_task(process_and_score_responses(uids))
        except Exception as e:
            bt.logging.error(f"Error in organic: {e}")
            raise e
