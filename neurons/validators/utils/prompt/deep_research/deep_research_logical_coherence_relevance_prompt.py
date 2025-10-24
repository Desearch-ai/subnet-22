from neurons.validators.utils.prompts import BasePrompt
from desearch.utils import call_openai
import re

user_template = """
Here's report data.
<Report>
{}
</Report>
"""

system_message = """
Scoring Guide

Role: As an evaluator, your task is to evaluate the logical coherence of a report data. You need to check if report data is following a logical flow, without contradictions and if it is backed by evidence or reasoning.
Follow these steps.

1. Input Information
-Report data: <Report>

2. Rules for checking logical coherence
- The report should start with "Introduction" and end with "Conclusion".
- Sections should flow logically, with related topics ordered properly (e.g., "Essential Components" before "Consensus Mechanisms").
- Subsections should align with their parent sections and follow a natural progression.
- Ensure smooth transitions between sections without abrupt shifts or gaps.
- Avoid redundant content within the same or across sections.

3. Output
Score 10:
- Criteria:
    - The report follows a perfect logical flow, with sections that are organized in a clear, coherent manner, and all topics are well-ordered and connected.
    - The report starts with an appropriate introduction and ends with a solid conclusion.
    - Transitions between sections are smooth and natural, with no abrupt shifts.
    - There is no redundancy, and all information is presented in a way that flows naturally from one idea to the next, without gaps.
    - All sections and subsections align well, supporting the overall narrative without contradictions or confusion.
- Example:
  Score 10: The report starts with a concise introduction, flows logically through well-organized sections and subsections, and ends with a strong conclusion. Transitions are smooth, and no contradictions or redundant content are present. Every section is perfectly aligned with the others.

Score 5:
- Criteria:
    - The report mostly follows a logical flow, but there are some minor issues that disrupt the flow (e.g., small gaps between sections, a few awkward transitions, or slight misordering of topics).
    - The report starts with an introduction and ends with a conclusion, but either the introduction or conclusion might lack sufficient clarity.
    - There might be some redundancy within sections or between sections, but it doesn’t significantly hinder understanding.
    - Small contradictions or minor misalignments between sections or subsections that do not seriously affect the overall coherence of the report.
- Example:
  Score 5: The report generally flows well, but there are a few instances of awkward transitions, minor redundancy, or slight contradictions between sections. However, these issues don’t completely disrupt the overall narrative, and the report is still understandable.

Score 2:
- Criteria:
    - The report has significant issues with its logical flow (e.g., major gaps between sections, disorganized structure, abrupt shifts in topics).
    - Sections and subsections may not align well with each other, and the report lacks smooth transitions between ideas.
    - There is considerable redundancy within the report, either within the same section or across sections.
    - Contradictions or inconsistencies are present in key areas, undermining the report's overall coherence and clarity.
    - The introduction and conclusion may be weak or unclear, leaving the reader confused about the report’s objectives and conclusions.
- Example:
  Score 2: The report has a lot of logical issues, such as sections that feel disjointed, major gaps in reasoning, or inconsistencies in the data presented. Redundancies and contradictions weaken the overall coherence, making it difficult for the reader to follow the narrative. The introduction and conclusion also lack clarity.
 
Output Format:
Score: [2, 5, or 10], Explanation:
"""


class DeepResearchLogicalCoherenceRelevancePrompt(BasePrompt):
    def __init__(self):
        super().__init__()
        self.template = user_template

    def get_system_message(self):
        return system_message

    async def get_response(self, report):
        return await call_openai(
            [
                {
                    "role": "system",
                    "content": self.get_system_message(),
                },
                {
                    "role": "user",
                    "content": self.text(report),
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
