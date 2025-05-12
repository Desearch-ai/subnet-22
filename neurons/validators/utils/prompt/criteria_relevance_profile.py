from neurons.validators.utils.prompts import BasePrompt
from datura.utils import call_openai
import re

user_template = """
<SearchCriteria>
{}
</SearchCriteria>

<UserProfile>
{}
</UserProfile>
"""

system_message = """
Relevance Scoring Guide:

Role: As an evaluator, your task is to determine how well a user profile relevant to search criteria.

Rules:
If user profile matches search criteria return 10.
If user profile doesn't match search criteria return 0.
Output with reason (what specific field of user profile) why profile match or doesn't match search criteria.

Output Format:
Score: [0, or 10], Explanation:
"""


class SearchCriteriaRelevancePrompt(BasePrompt):
    def __init__(self):
        super().__init__()
        self.template = user_template

    def get_system_message(self):
        return system_message

    async def get_response(self, criteria, profile):
        return await call_openai(
            [
                {
                    "role": "system",
                    "content": self.get_system_message(),
                },
                {
                    "role": "user",
                    "content": self.text(criteria, profile),
                },
            ],
            temperature=0.1,
            model="gpt-4o-mini",
        )

    def extract_explanation(self, response: str) -> str:
        match = re.search(r"(?<=Explanation:\s)(.*)", response)

        if match:
            return match.group(1)

        return ""

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
