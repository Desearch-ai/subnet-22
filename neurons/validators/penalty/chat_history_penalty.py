import torch
from typing import List
from neurons.validators.utils.tasks import Task
from neurons.validators.penalty.penalty import BasePenaltyModel, PenaltyModelType
import bittensor as bt
from datura.protocol import ChatHistoryItem, ScraperStreamingSynapse, ScraperTextRole
from neurons.validators.utils.prompt.chat_history_relevance_prompt import (
    ChatHistoryRelevancePrompt,
)


class ChatHistoryPenaltyModel(BasePenaltyModel):
    @property
    def name(self) -> str:
        return PenaltyModelType.chat_history_penalty.value

    async def validate_summary_with_rule(
        self, completion: str, chat_history: List[ChatHistoryItem], prompt: str
    ):
        summary_rule_prompt = ChatHistoryRelevancePrompt()

        chat_history_text = "\n".join(
            f"<ChatHistoryItem>\n<Prompt><{item.prompt}></Prompt>\n<Completion><{item.completion}</Completion>\n</ChatHistoryItem>"
            for item in chat_history
        )

        response = await summary_rule_prompt.get_response(
            completion, chat_history_text, prompt
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
            if not response.chat_history:
                penalties[index] = 0.0
                bt.logging.debug(
                    f"Response index {index} has no chat_history. No penalty."
                )
                continue

            if (
                await self.validate_summary_with_rule(
                    response.texts["summary"], response.chat_history, response.prompt
                )
                < 10
            ):
                penalties[index] = self.max_penalty

            bt.logging.debug(f"Response index {index} has penalty {penalties[index]}")

        return penalties
