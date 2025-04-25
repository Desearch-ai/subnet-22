from neurons.validators.utils.prompts import BasePrompt
from datura.utils import call_openai
import re

user_template = """
Here's user profile information.
<UserProfile>
{}
</UserProfile>

Here's search criterion.
<Criterion>
{}
</Criterion>

Here's criterion summary.
<CriterionSummary>
{}
</CriterionSummary>
"""

system_message = """
Scoring Guide

Role: As an evaluator, your task is to evaluate the adherence of a generated criterion summary. You need to check if summary is correctly representing the profile based on the criterion.
Follow these steps.

1. Input Information
-User profile information: <UserProfile>
-Criterion: <Criterion>
-Generated Criterion Summary: <CriterionSummary>

2. Evaluation Criteria
- Check if the criterion summary is based on user profile.
- Check if the criterion summary is summary of criterion provided.

3. Output
- Assign a score based on adherence:
  - Score 10: If the summary fully adheres to the criterion and user profile
  - Score 0: If the summary does not adhere to the criterion and user profile.

- Output the score and the reason:
  Example output
  - Score [0, or 10]: Explanation
"""


class SearchCriteriaSummaryRelevancePrompt(BasePrompt):
    def __init__(self):
        super().__init__()
        self.template = user_template

    def get_system_message(self):
        return system_message

    async def get_response(self, user_profile, criterion, criterion_summary):
        return await call_openai(
            [
                {
                    "role": "system",
                    "content": self.get_system_message(),
                },
                {
                    "role": "user",
                    "content": self.text(user_profile, criterion, criterion_summary),
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
