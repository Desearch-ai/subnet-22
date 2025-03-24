import torch
from typing import List
from neurons.validators.utils.tasks import Task
from neurons.validators.penalty.penalty import BasePenaltyModel, PenaltyModelType
import bittensor as bt
from datura.protocol import PeopleSearchSynapse
from neurons.validators.utils.prompt.search_criteria_relevance import (
    SearchCriteriaRelevancePrompt,
)
from datura.utils import str_linkedin_profile
from datura.synapse import collect_responses

MAX_PENALTY = 1.0


class CriteriaSummaryPenaltyModel(BasePenaltyModel):
    def __init__(self, max_penalty: float = MAX_PENALTY):
        super().__init__(max_penalty)
        bt.logging.debug(
            "Initialized CriteriaSummaryPenaltyModel using max_execution_time from responses."
        )

    @property
    def name(self) -> str:
        return PenaltyModelType.criteria_summary_penalty.value

    async def validate_summary(self, user_profile, criterion, criterion_summary):
        search_criteria_relevance_prompt = SearchCriteriaRelevancePrompt()

        response = await search_criteria_relevance_prompt.get_response(
            user_profile, criterion, criterion_summary
        )

        return search_criteria_relevance_prompt.extract_score(response)

    async def calculate_penalties(
        self,
        responses: List[PeopleSearchSynapse],
        tasks: List[Task],
        additional_params=None,
    ) -> torch.FloatTensor:

        penalties = torch.zeros(len(responses), dtype=torch.float32)

        for index, response in enumerate(responses):
            scores = []
            async_actions = []
            for result in response.results:
                for i, criterion in enumerate(response.criteria):

                    async def calculate_score(result, criterion, i):
                        try:
                            criterion_summary = result.get("criteria_summary")

                            if not criterion_summary:
                                scores.append(0.0)
                                return

                            if (
                                await self.validate_summary(
                                    str_linkedin_profile(result),
                                    criterion,
                                    criterion_summary[i],
                                )
                                < 10
                            ):
                                scores.append(0.0)
                                return

                            scores.append(1.0)
                        except:
                            scores.append(0.0)

                    async_actions.append(calculate_score(result, criterion, i))

            await collect_responses(async_actions)

            penalties[index] = 1 - (sum(scores) / len(scores) if scores else 0.0)
            bt.logging.debug(f"Response index {index} has penalty {penalties[index]}")

        return penalties
