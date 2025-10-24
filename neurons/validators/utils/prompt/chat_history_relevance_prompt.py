from neurons.validators.utils.prompts import BasePrompt
from desearch.utils import call_openai
import re

user_template = """
Here is completion text:
<Completion>
{}
</Completion>

And the chat history:
<ChatHistory>
{}
</ChatHistory>

And the current prompt:
<CurrentPrompt>
{}
</CurrentPrompt>
"""

system_message = """
Scoring Guide

Role: Your role is to evaluate whether a generated completion accurately reflects the information from a chat history. Follow these steps:
Follow these steps.

1. Input Information
-Chat history: <ChatHistory>
-Generated Completion: <Completion>
-Current Prompt: <CurrentPrompt>

2. Evaluation Criteria
- Check if the completion is derived from and consistent with the chat history.
- Check if it fairly represents the key details, context, and meaning from the conversation.

3. Output
- Assign a score based on adherence:
  - Score 10: If the completion fully and accurately reflects the chat history
  - Score 0: If the completion is missing, incorrect, or unrelated to the chat history

- Output the score and the reason:
  Example output
  - Score [0, or 10]: Explanation
"""


class ChatHistoryRelevancePrompt(BasePrompt):
    def __init__(self):
        super().__init__()
        self.template = user_template

    def get_system_message(self):
        return system_message

    async def get_response(self, completion, chat_history, prompt):
        return await call_openai(
            [
                {
                    "role": "system",
                    "content": self.get_system_message(),
                },
                {
                    "role": "user",
                    "content": self.text(completion, chat_history, prompt),
                },
            ],
            temperature=0.8,
            model="gpt-4o-mini",
        )

    def extract_score(self, response: str) -> float:
        r"""Extract numeric score (range 0-10) from prompt response."""
        # Mapping of special codes to numeric scores

        # Extract score from output string with various formats
        match = re.search(r"(?i)score[:\s]*(\d+)", response)
        if match:
            try:
                score = float(match.group(1))
                if 0 <= score <= 10:
                    return score
            except ValueError:
                return 0

        # Extract score directly from the response if "Score:" prefix is missing
        match = re.search(r"\b(\d+)\b", response)
        if match:
            try:
                score = float(match.group(1))
                if 0 <= score <= 10:
                    return score
            except ValueError:
                return 0

        return 0
