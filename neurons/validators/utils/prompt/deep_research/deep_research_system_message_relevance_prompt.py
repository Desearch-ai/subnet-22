from neurons.validators.utils.prompts import BasePrompt
from desearch.utils import call_openai
import re

user_template = """
Here's Report data.
<Report>
{}
</Report>

Here's user system message
<UserSystemMessage>
{}
</UserSystemMessage>
"""

system_message = """
Scoring Guide

Role: As an evaluator, your task is to evaluate the adherence of a report to specific guidelines outlined in a user system message.
Follow these steps.

1. Input Information
-Report data: <Report>
-User system message: <UserSystemMessage>

Before evaluating, caculate the numbers that is required on the user system message (e.g. section count, sentence count in each section, ...) and write what you have calculated.

**
Some assumes about the user system message.
When it says section in the user system message, do not care about subsection. It's only referring to main section.
**

2. Evaluation Criteria
- Check if the report captures the main points outlined in the user system message.
- Verify that the tone and style of the report description match the specifications in the user system message
- Assess whether the report description maintains the required length and structure as indicated in the user system message
- Look for any specific keywords or phrases mentioned in the user system message and confirm their presence in the report.
- *When checking the number and count (e.g. section count, sentence count), return high score if the numbers are similar.*

3. Output
Score 10:
- Criteria:
    - The report perfectly captures all the main points outlined in the user system message.
    - The tone and style of the report exactly match the guidelines provided in the user system message.
    - The report adheres to the specified length and structure, and no required elements are missing.
    - The report contains all the specific keywords or phrases mentioned in the user system message, and their use is appropriate.
    - The report is fully aligned with the user's expectations based on the guidelines.
- Example:
  Score 10: The report perfectly adheres to the user system message. It captures all main points, maintains the correct tone, follows the required structure and length, and includes all necessary keywords/phrases.

Score 5:
- Criteria:
    - The report mostly captures the main points, but there are some minor omissions or slight deviations from the user system message.
    - The tone and style are generally in line with the specifications, but there may be a few inconsistencies.
    - The report is close to the required length and structure, but there may be minor deviations.
    - Some keywords or phrases from the user system message are present, but a few might be missing or used incorrectly.
- Example:
  Score 5: The report covers the key points and aligns with the user message in most areas, but some minor aspects (tone, length, keywords) need slight adjustments to fully match the guidelines.

Score 2:
- Criteria:
    - The report captures only a few of the main points or includes significant omissions from the user system message.
    - The tone and style are inconsistent or partially off from the required specifications.
    - The report has length or structural issues that significantly differ from the user system message guidelines.
    - Many of the specific keywords or phrases mentioned in the user system message are either missing or incorrectly used.
- Example:
  Score 2: The report misses several key points, has noticeable issues with tone/style or structure, and lacks most of the keywords/phrases from the user system message.

Output Format:
Score: [2, 5, or 10], Explanation:


** Must start with Score: **
"""


class DeepResearchSystemMessageRelevancePrompt(BasePrompt):
    def __init__(self):
        super().__init__()
        self.template = user_template

    def get_system_message(self):
        return system_message

    async def get_response(self, report, user_system_message):
        return await call_openai(
            [
                {
                    "role": "system",
                    "content": self.get_system_message(),
                },
                {
                    "role": "user",
                    "content": self.text(report, user_system_message),
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
