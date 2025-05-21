import torch
from typing import List
from neurons.validators.utils.tasks import Task
from neurons.validators.penalty.penalty import BasePenaltyModel, PenaltyModelType
import bittensor as bt
from datura.protocol import DeepResearchSynapse
from neurons.validators.utils.prompt.deep_research.deep_ressearch_stream_check_prompt import (
    DeepResearchStreamCheckPrompt,
)
import statistics

MAX_TOKENS_PER_CHUNK = 2
PENALTY_PER_EXCEEDING_TOKEN = 0.01


class DeepResearchStreamingPenaltyModel(BasePenaltyModel):
    @property
    def name(self) -> str:
        return PenaltyModelType.streaming_penalty.value

    async def check_logic_order(self, response: DeepResearchSynapse):
        try:
            prompt = DeepResearchStreamCheckPrompt()

            response = await prompt.get_response(
                response.flow_items.__str__(),
                response.prompt,
                response.system_message or "",
            )
            print("AAAa", response)
            score = prompt.extract_score(response) / 10

            return 1 - score
        except:
            return 1

    def check_stream_times(self, response: DeepResearchSynapse, threshold: int = 5):
        try:
            items = response.flow_items

            if not items or len(items) < 2:
                return 1

            times = [item.time for item in items]
            time_range = max(times) - min(times)

            if time_range <= threshold:
                return 1

            return 0
        except:
            return 1

    async def calculate_penalties(
        self,
        responses: List[DeepResearchSynapse],
        tasks: List[Task],
        additional_params=None,
    ) -> torch.FloatTensor:
        accumulated_penalties = torch.zeros(len(responses), dtype=torch.float32)

        for index, response in enumerate(responses):
            scores = []
            scores.append(await self.check_logic_order(response))
            scores.append(self.check_stream_times(response))

            accumulated_penalties[index] = sum(scores) / len(scores)

        return accumulated_penalties
