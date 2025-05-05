from neurons.validators.utils.prompts import BasePrompt
from datura.utils import call_openai
import re

user_template = """
Here's source links data.
<LinksData>
{}
</LinksData>

Here's users' prompt.
<UserPrompt>
{}
</UserPrompt>
"""

system_message = """
Scoring Guide

Role: As an evaluator, your task is to evaluate the adherence of a source links. You need to check if source links and it's web content data is related to user's prompt. If it doesn't contain any information related to user's prompt it should return 2.
Follow these steps.

1. Input Information
-User prompt: <UserPrompt>
-Source links and it's data: <LinksData>

2. Evaluation Criteria
- The source link must contain content directly related to the user's prompt or topic. If the source does not address the prompt, return 2.
- Ensure the information on the linked webpage aligns with the subject matter of the prompt. Verify that the source provides meaningful insights or data.
- The source must provide accurate and reliable information relevant to the user's query. If the content is irrelevant or misleading, return 2.
- Avoid sources that provide tangential or unrelated content. If the source deviates significantly from the prompt, return 2.
- The source should ideally be up-to-date, especially if the topic is sensitive to changes over time. If the source is outdated or lacks recent information, consider it irrelevant.
- The source should not be overly general or superficial. It should provide specific insights or data that directly help address the user's request.

3. Output
Score 10:
- Criteria:
    - The source link provides directly relevant content that addresses the user's prompt in detail.
    - The information on the linked webpage is accurate, reliable, and aligns perfectly with the subject matter of the user's request.
    - The source is up-to-date, and the content is specific, providing meaningful insights that directly help answer the query.
    - The source doesn't veer off-topic and offers precise data or analysis that contributes significantly to the user's understanding of the prompt.
- Example:
  Score 10: The source link contains accurate, specific, and up-to-date information that is directly aligned with the user's query. The content is detailed and informative, helping to comprehensively answer the prompt without deviation.

Score 5:
- Criteria:
    - The source link provides content that is partially related to the user's prompt, but it may not fully address the query or leaves some gaps in coverage.
    - The source contains some relevant insights or data, but there may be parts of the webpage that are less directly related or slightly off-topic.
    - The content is generally accurate, but some minor discrepancies or outdated information may be present.
    - The source is useful, but it doesn't fully satisfy the prompt or could have been more specific.
- Example:
  Score 5: The source is somewhat relevant to the user's query, but it may cover the topic only partially or not in enough detail. Some sections might not fully align with the query, but it still provides meaningful insights.

Score 2:
- Criteria:
    - The source link contains some relevant content, but much of it is irrelevant, tangential, or superficial.
    - The source may provide general information but does not deeply address the user's prompt or offer specific insights.
    - The content is mostly not aligned with the user's request, with only small portions potentially related.
    - There are significant gaps in the information, or the source doesn't directly help answer the user's query.
- Example:
  Score 2: The source provides some vaguely related content but veers off-topic in many areas. The relevance to the prompt is limited, and the source doesn't help answer the user's request in a meaningful way.
 
Output Format:
Score: [2, 5, or 10], Explanation:
"""


class DeepResearchSourceLinksRelevancePrompt(BasePrompt):
    def __init__(self):
        super().__init__()
        self.template = user_template

    def get_system_message(self):
        return system_message

    async def get_response(self, source_links_data, user_prompt):
        return await call_openai(
            [
                {
                    "role": "system",
                    "content": self.get_system_message(),
                },
                {
                    "role": "user",
                    "content": self.text(source_links_data, user_prompt),
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
