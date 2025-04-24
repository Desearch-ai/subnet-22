from neurons.validators.utils.prompts import BasePrompt
from datura.utils import call_openai
import re

user_template = """
Here's description data.
<Description>
{}
</Description>

Here's web data.
<WebData>
{}
</WebData>
"""

system_message = """
Scoring Guide

Role: As an evaluator, your task is to evaluate the correctness of informations(figures, names, dates ...) provided in description based on the web data provided.
Follow these steps.

1. Input Information
-Description data: <Description>
-Web data: <WebData>

2. Evaluation Criteria
- Verify that any numerical data (e.g., statistics, quantities, measurements) in the description matches the web source. If discrepancies exist, flag them as incorrect.
- Check that any names (e.g., people, companies, organizations) in the description are correctly spelled and correspond to the right entities as verified from the web source.
- Ensure that any dates (e.g., event dates, founding years, publication dates) in the description are consistent with the dates found on the web source. Flag any incorrect or mismatched dates.
- Cross-reference all information (facts, events, claims) in the description with the web data to ensure they are consistent. If any claim doesn't align with the web data, mark it as incorrect.
- Verify that the context in which figures, names, and dates are used is accurate. Ensure they are not taken out of context or misrepresented.
- Identify and flag any misleading or false information presented in the description, even if it seems correct at first glance.
- If no web data is provided, just check if description is correct normally.

3. Output
Score 10:
- Criteria:
    - All figures, names, dates, and facts in the description are 100% correct, matching the web source perfectly, with no discrepancies or misleading information.
    - The context of the information is completely accurate, and no false claims are made.
    - The description aligns exactly wit hthe web data and provide clear, precise, and factual information.
- Example:
  Score 10: The description perfectly aligns with the web data. All dates, names, and figures are correct, and the context is accurately represented. No discrepancies or false claims were found.

Score 5:
- Criteria:
    - Some minor inaccuracies or discrepancies exist in the description (e.g., small spelling mistakes, slightly incorrect numbers, or slightly off dates).
    - The core information remains mostly correct, but a few details need minor correction.
    - The context is largely intact, but one or two small misrepresentations may be present, although they don’t significantly affect the overall understanding.
- Example:
  Score 5: The description is generally correct, but a few dates or figures do not fully align with the web data. While the context is mostly right, small discrepancies were found, but they don’t change the overall message.

Score 2:
- Criteria:
    - The description contains significant inaccuracies or mismatches with the web data.
    - A considerable amount of information (names, dates, numbers) is incorrect or incomplete, even if it seems like it might be correct initially.
    - The description may be misleading or out of context, causing potential confusion or misunderstanding of key facts.
    - The core structure of the description is largely wrong or misrepresented.
- Example:
  Score 2: The description contains several incorrect dates, figures, or names that do not align with the web source. Additionally, the context is misleading or false in several areas, and the overall accuracy is compromised.
 

Output Format:
Score: [2, 5, or 10], Explanation:
"""


class DeepResearchDataRelevancePrompt1(BasePrompt):
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
