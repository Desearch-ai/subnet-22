from neurons.validators.utils.prompts import BasePrompt
from datura.utils import call_openai
import re

user_template = """
Here's section data.
<SectionData>
{}
</SectionData>

Here's users' prompt.
<UserPrompt>
{}
</UserPrompt>
"""

system_message = """
Scoring Guide

Role: As an evaluator, your task is to evaluate the adherence of a section of report data. You need to check if the section data correctly answers or addresses the user's prompt.
Follow these steps.

1. Input Information
-User prompt: <UserPrompt>
-Section data: <SectionData>

2. Evaluation Criteria
- Check if the section data contributes to answering or addressing the user’s prompt, even if it doesn’t directly provide a complete answer.
- Ensure that the section data provides relevant information related to the overall topic of the prompt. It may not directly answer the question but should still contribute to understanding the topic.
- Verify that the section is complete in its coverage of the topic it addresses, ensuring that it doesn’t leave out key details, and that the content is clear and easy to follow.
- Ensure the section provides accurate information, and that it is contextually appropriate to the prompt, even if it isn’t directly answering it.

3. Output
Score 10:
- Criteria:
    - The section data provides relevant, accurate, and clear information that contributes meaningfully to the topic of the user's prompt.
    - The content in the section may not directly answer the prompt but still provides valuable context or background that helps the user understand the topic as a whole.
    - The section is complete, coherent, and adds important insights without deviating into irrelevant areas.

- Example:
  Score 10: The section provides relevant and accurate details related to blockchain, even if it doesn’t directly answer “What is blockchain?”. It could discuss aspects like its applications or technology, contributing important context to the prompt.

Score 5:
- Criteria:
    - The section data provides some relevance to the prompt, but may be too general or only partially related to the user's query.
    - The information is generally aligned with the topic but lacks completeness or goes off-topic in some parts.
    - The section may provide useful background or secondary information, but doesn’t fully cover all necessary details to answer the prompt.

- Example:
  Score 5: The section provides useful information about blockchain but might be too focused on one specific application or only cover part of the blockchain concept. It adds some value but doesn’t fully address the user’s query or other aspects of the topic.

Score 2:
- Criteria:
    - The section data has minimal relevance to the prompt and does not contribute effectively to answering the question or understanding the topic.
    - There are significant gaps in the content, and much of the information included is irrelevant or off-topic.
    - The section is incomplete or unclear in its treatment of the topic, providing little value for the user’s understanding of the subject.

- Example:
  Score 2: The section provides information that is only tangentially related to blockchain, such as an unrelated application or concept, which doesn’t meaningfully contribute to answering the user’s prompt or provide helpful context.

Output Format:
Score: [2, 5, or 10], Explanation: Provide an explanation of why the section data was scored, referencing how well it contributes to answering or addressing the user’s prompt, even if it doesn’t provide a direct answer.
"""


class DeepResearchContentRelevancePrompt(BasePrompt):
    def __init__(self):
        super().__init__()
        self.template = user_template

    def get_system_message(self):
        return system_message

    async def get_response(self, report, user_prompt):
        return await call_openai(
            [
                {
                    "role": "system",
                    "content": self.get_system_message(),
                },
                {
                    "role": "user",
                    "content": self.text(report, user_prompt),
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
