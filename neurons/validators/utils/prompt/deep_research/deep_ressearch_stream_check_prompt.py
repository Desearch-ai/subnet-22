from neurons.validators.utils.prompts import BasePrompt
from datura.utils import call_openai
import re

user_template = """
Here's stream data.
<StreamData>
{}
</StreamData>

Here's users' prompt.
<UserPrompt>
{}
</UserPrompt>

Here's user system message
<UserSystemMessage>
{}
</UserSystemMessage>
"""

system_message = """
Scoring Guide
Role: As an evaluator, your task is to evaluate the logical order of streamed thinking steps during a deep research process. You need to determine whether the steps follow a natural and coherent progression, based on the user’s prompt and system instructions. If the steps are disorganized, skip essential phases, or make the reasoning unclear, return a low score.

Follow these steps.

1. Input Information
    -User prompt: <UserPrompt>
    -User system message: <UserSystemMessage>
    -Streamed step data: <StreamData>

2. Evaluation Criteria
    The thinking steps must follow a logical flow — typically starting with discovery or search, followed by reading or analysis, and finishing with a conclusion or summary.
    All major phases of deep research should be present and in order: searching → analyzing → summarizing.
    The language used in the steps should be understandable and clearly convey what action the miner is performing.
    Even if different miners phrase or structure their steps differently, the overall reasoning must be consistent and easy to follow.
    Minor variations are acceptable, but large jumps, random sequences, or missing steps should reduce the score.

3. Output
Score 10:
    Criteria:
    - The stream clearly includes all major phases in a natural and logical order.
    - The steps are well-written and easy to follow.
    - The miner's reasoning makes sense from start to finish.

Example:
    Score 10: The stream begins with search and discovery, continues with reading and evaluation of sources, and finishes with a conclusion. The steps follow a clear logical structure that matches how humans think through research.

Score 5:
    Criteria:
    - The stream is mostly logical, but one or two steps feel out of place or unclear.
    - Some transitions may be weak, but the overall reasoning is still understandable.
    - One phase might be missing or lightly represented, but the sequence mostly makes sense.

Example:
    Score 5: The stream follows a mostly correct order, but there’s a confusing jump between steps or a missing phase. It’s still helpful, but not entirely clean or clear.

Score 2:
    Criteria:
    - The steps appear random or confusing.
    - Major phases of research are skipped or completely out of order.
    - It’s hard to understand the miner’s reasoning or how the final output was reached.

Example:
    Score 2: The stream jumps from summarizing to searching, then back to evaluating. The logic is unclear, and the steps do not follow a natural flow of thought.

Output Format:
    Score: [2, 5, or 10], Explanation:
"""


class DeepResearchStreamCheckPrompt(BasePrompt):
    def __init__(self):
        super().__init__()
        self.template = user_template

    def get_system_message(self):
        return system_message

    async def get_response(self, flow_items, user_prompt, user_system_message):
        return await call_openai(
            [
                {
                    "role": "system",
                    "content": self.get_system_message(),
                },
                {
                    "role": "user",
                    "content": self.text(flow_items, user_prompt, user_system_message),
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
