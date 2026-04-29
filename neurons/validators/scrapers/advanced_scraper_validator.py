import time
from typing import List, Optional

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
from neurons.validators.clients.miner_response_logger import (
    build_log_entry,
    submit_logs_best_effort,
)
from neurons.validators.penalty.miner_score_penalty import MinerScorePenaltyModel
from neurons.validators.penalty.streaming_penalty import StreamingPenaltyModel
from neurons.validators.penalty.timeout_penalty import TimeoutPenaltyModel
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
from neurons.validators.scoring import capacity
from neurons.validators.scrapers.base_scraper_validator import BaseScraperValidator

TWITTER_TOOL = "Twitter Search"
WEB_TOOLS = frozenset(
    [
        "Web Search",
        "Wikipedia Search",
        "Youtube Search",
        "ArXiv Search",
        "Reddit Search",
        "Hacker News Search",
    ]
)


class AdvancedScraperValidator(BaseScraperValidator):
    search_type = "ai_search"
    wandb_modality = "twitter_scrapper"
    wandb_reward_keys = ["twitter_reward", "search_reward", "summary_reward"]

    def __init__(self, neuron: AbstractNeuron):
        self.language = "en"
        self.region = "us"
        self.date_filter = "qdr:w"  # Past week

        # Init device.
        bt.logging.debug("loading", "device")
        bt.logging.debug(
            "self.neuron.config.neuron.device = ", str(neuron.config.neuron.device)
        )

        self.twitter_content_weight = 0.30
        self.web_search_weight = 0.25
        self.summary_relevance_weight = 0.25
        self.performance_weight = 0.20

        self.reward_llm = RewardLLM(neuron.config.neuron.scoring_model)

        reward_weights = torch.tensor(
            [
                self.twitter_content_weight,
                self.web_search_weight,
                self.summary_relevance_weight,
                self.performance_weight,
            ],
            dtype=torch.float32,
        )

        reward_functions = [
            TwitterContentRelevanceModel(
                device=neuron.config.neuron.device,
                scoring_type=RewardScoringType.summary_relevance_score_template,
                llm_reward=self.reward_llm,
                neuron=neuron,
            ),
            WebSearchContentRelevanceModel(
                device=neuron.config.neuron.device,
                scoring_type=RewardScoringType.search_relevance_score_template,
                llm_reward=self.reward_llm,
                neuron=neuron,
            ),
            SummaryRelevanceRewardModel(
                device=neuron.config.neuron.device,
                scoring_type=RewardScoringType.summary_relevance_score_template,
                llm_reward=self.reward_llm,
                neuron=neuron,
            ),
            PerformanceRewardModel(
                device=neuron.config.neuron.device,
                neuron=neuron,
                min_realistic_time=5.0,
                target_time=10.0,
            ),
        ]

        penalty_functions = [
            StreamingPenaltyModel(max_penalty=1, neuron=neuron),
            TimeoutPenaltyModel(max_penalty=1, neuron=neuron),
            MinerScorePenaltyModel(max_penalty=1, neuron=neuron),
        ]

        super().__init__(
            neuron=neuron,
            reward_weights=reward_weights,
            reward_functions=reward_functions,
            penalty_functions=penalty_functions,
        )

    def compute_reward_weights_matrix(self, responses) -> torch.Tensor:
        rows = [self._weights_for(response) for response in responses]
        return torch.tensor(rows, dtype=torch.float32)

    def _weights_for(self, response) -> List[float]:
        """Weights for one response, ordered [twitter, web, summary, perf].

        Content weight is split between twitter and web only when both tool
        categories are used. If only one is used, it absorbs the other's share
        — so the unused branch gets weight 0 rather than the old `reward = 1.0`
        free pass.
        """
        tools = set(response.tools)
        has_twitter = TWITTER_TOOL in tools
        has_web = bool(tools & WEB_TOOLS)
        content = self.twitter_content_weight + self.web_search_weight

        if has_twitter and has_web:
            w_twitter, w_web = self.twitter_content_weight, self.web_search_weight
        elif has_twitter:
            w_twitter, w_web = content, 0.0
        else:
            w_twitter, w_web = 0.0, content

        return [
            w_twitter,
            w_web,
            self.summary_relevance_weight,
            self.performance_weight,
        ]

    async def _dendrite_stream(
        self,
        synapse: ScraperStreamingSynapse,
        axon,
        uid: int,
        timeout: float,
    ):
        """Wrap ``dendrite.call_stream`` so chunks flow through to the caller
        and per-call success is recorded once the stream ends."""
        dendrite = next(self.neuron.dendrites)
        final_synapse = None
        success = False
        try:
            async for value in dendrite.call_stream(
                target_axon=axon,
                synapse=synapse,
                timeout=timeout,
                deserialize=False,
            ):
                if isinstance(value, bt.Synapse):
                    final_synapse = value
                yield value
            status = getattr(
                getattr(final_synapse, "dendrite", None), "status_code", None
            )
            success = status == 200
        except Exception as e:
            bt.logging.error(f"[{self.search_type}] dendrite stream failed uid={uid}: {e}")

        await capacity.note_call_result(uid, self.search_type, success)

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

        uid, axon = await self.neuron.get_random_miner(
            uid=uid, search_type=self.search_type
        )
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

        async_response = self._dendrite_stream(
            synapse.model_copy(),
            axon,
            uid,
            timeout=max_execution_time + 5,
        )

        return async_response, uids, start_time, axon

    def get_penalty_additional_params(self, val_score_responses_list):
        val_scores = []
        for val_score_responses, reward_function in zip(
            val_score_responses_list, self.reward_functions
        ):
            if reward_function.name in [
                RewardModelType.twitter_content_relevance.value,
                RewardModelType.search_content_relevance.value,
            ]:
                val_scores.append(val_score_responses)
        return val_scores

    def build_uid_log_message(self, uid, reward, response):
        completion_length = (
            len(response.completion) if response.completion is not None else 0
        )
        bt.logging.trace(f"{response.completion}")
        return f"UID: {uid}, R: {round(reward, 3)}, C: {completion_length}"

    def populate_wandb_uid_data(self, wandb_data, uid, reward, response, reward_values):
        wandb_data["scores"][uid] = reward
        wandb_data["responses"][uid] = response.completion
        wandb_data["prompts"][uid] = response.prompt
        for key, value in zip(self.wandb_reward_keys, reward_values):
            wandb_data[key][uid] = value

    async def send_scoring_query(
        self,
        query: dict,
        uid: int,
    ) -> Optional[object]:
        """Send a scoring query to a specific miner via dendrite (streaming).
        Consumes the stream and returns the final populated synapse."""
        prompt = query["query"]
        tools = query.get("tools", [])
        date_filter = get_specified_date_filter(
            DateFilterType(
                query.get("date_filter_type", DateFilterType.PAST_WEEK.value)
            )
        )

        max_execution_time = get_max_execution_time(Model.NOVA, 10)

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

        synapse = ScraperStreamingSynapse(
            prompt=prompt,
            model=Model.NOVA,
            result_type=ResultType.LINKS_WITH_FINAL_SUMMARY,
            start_date=start_date,
            end_date=end_date,
            date_filter_type=(
                date_filter.date_filter_type.value
                if date_filter.date_filter_type
                else None
            ),
            tools=tools,
            language=self.language,
            region=self.region,
            google_date_filter=self.date_filter,
            max_execution_time=max_execution_time,
            scoring_model=self.neuron.config.neuron.scoring_model,
        )

        axon = self.neuron.metagraph.axons[uid]
        final_synapse = None
        async for value in self._dendrite_stream(
            synapse,
            axon,
            uid,
            timeout=max_execution_time + 5,
        ):
            if isinstance(value, bt.Synapse):
                final_synapse = value
        return final_synapse

    async def organic(
        self,
        query,
        model: Optional[Model] = Model.NOVA,
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

            async_response, uids, start_time, axon = await self.call_miner(
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
                chat_history=chat_history,
                count=count,
            )

            final_synapses = []
            selected_uid = uids[0].item() if len(uids) else None

            if is_collect_final_synapses:
                # Collect specified uids from responses and score
                final_synapses = await collect_final_synapses(
                    [async_response], uids, start_time
                )

                if final_synapses:
                    submit_logs_best_effort(
                        self.neuron,
                        [
                            build_log_entry(
                                owner=self.neuron,
                                search_type="ai_search",
                                query_kind="organic",
                                response=synapse,
                                miner_uid=selected_uid,
                                miner_hotkey=getattr(axon, "hotkey", None),
                                miner_coldkey=getattr(axon, "coldkey", None),
                            )
                            for synapse in final_synapses
                            if synapse is not None
                        ],
                    )
                    for synapse in final_synapses:
                        if synapse is not None:
                            await self._save_organic_for_scoring(
                                uid=selected_uid, response=synapse
                            )

                for synapse in final_synapses:
                    yield synapse
            else:
                # Stream miner response to the UI
                final_synapse = None
                async for value in async_response:
                    if isinstance(value, bt.Synapse):
                        final_synapse = value
                    else:
                        yield value

                if final_synapse is not None:
                    submit_logs_best_effort(
                        self.neuron,
                        [
                            build_log_entry(
                                owner=self.neuron,
                                search_type="ai_search",
                                query_kind="organic",
                                response=final_synapse,
                                miner_uid=selected_uid,
                                miner_hotkey=getattr(axon, "hotkey", None),
                                miner_coldkey=getattr(axon, "coldkey", None),
                            )
                        ],
                    )
                    await self._save_organic_for_scoring(
                        uid=selected_uid, response=final_synapse
                    )
        except Exception as e:
            bt.logging.error(f"Error in organic: {e}")
            raise e
