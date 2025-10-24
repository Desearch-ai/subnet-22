import torch
from typing import List
from neurons.validators.base_validator import AbstractNeuron
from neurons.validators.utils.tasks import Task
from neurons.validators.penalty.penalty import BasePenaltyModel, PenaltyModelType
import bittensor as bt
from desearch.protocol import ScraperStreamingSynapse, ScraperTextRole
from desearch.utils import call_openai
from neurons.validators.utils.prompts import SummaryRulePrompt

MAX_PENALTY = 1.0

SUMMARIES = [
    ScraperTextRole.TWITTER_SUMMARY,
    ScraperTextRole.SEARCH_SUMMARY,
    ScraperTextRole.HACKER_NEWS_SUMMARY,
    ScraperTextRole.REDDIT_SUMMARY,
    ScraperTextRole.FINAL_SUMMARY,
]


class SummaryRulePenaltyModel(BasePenaltyModel):
    def __init__(self, max_penalty: float = MAX_PENALTY, neuron: AbstractNeuron = None):
        super().__init__(max_penalty, neuron)
        bt.logging.debug(
            "Initialized SummaryRulePenaltyModel using max_execution_time from responses."
        )

    @property
    def name(self) -> str:
        return PenaltyModelType.summary_rule_penalty.value

    async def validate_summary_with_rule(self, summary_text, summary_rule):
        summary_rule_prompt = SummaryRulePrompt()

        response = await call_openai(
            messages=summary_rule_prompt.get_messages(summary_text, summary_rule),
            temperature=0.0001,
            model="gpt-4o-mini",
        )

        return summary_rule_prompt.extract_score(response)

    async def calculate_penalties(
        self,
        responses: List[ScraperStreamingSynapse],
        tasks: List[Task],
        additional_params=None,
    ) -> torch.FloatTensor:

        penalties = torch.zeros(len(responses), dtype=torch.float32)

        for index, response in enumerate(responses):
            if not response.system_message:
                penalties[index] = 0.0
                bt.logging.debug(
                    f"Response index {index} has no system_message. No penalty."
                )
                continue

            for summary in SUMMARIES:
                chunks = response.text_chunks.get(summary)

                if not chunks:
                    continue

                summary_text = "".join(chunks)

                if (
                    await self.validate_summary_with_rule(
                        summary_text, response.system_message
                    )
                    < 10
                ):
                    penalties[index] += 1.0 / len(SUMMARIES)

            bt.logging.debug(f"Response index {index} has penalty {penalties[index]}")

        return penalties
