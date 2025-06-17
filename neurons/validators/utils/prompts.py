# The MIT License (MIT)
# Copyright © 2023 Yuma Rao
# Copyright © 2023 Opentensor Foundation

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish,pvali distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import re
import random
from typing import List, Optional
import json
from datura.protocol import ResultType, ScraperTextRole


class BasePrompt:
    r"""Base class for prompts expecting an extractable response."""

    def __init__(self):
        self.template = ""
        self.extract_pattern = ""

    def text(self, *args) -> str:
        r"""Sanitize input strings and format prompt template."""
        sanitized = args
        tags = find_unique_tags(self.template)
        for tag in tags:
            sanitized = [arg.replace(tag, "") for arg in sanitized]

        return self.template.format(*sanitized)

    def extract(self, response: str):
        r"""Search for the extract pattern in the text using regex."""
        result_pattern = re.compile(self.extract_pattern, re.DOTALL)
        result = re.findall(result_pattern, response)

        # If result found, return it.
        if result:
            return result[0]

        # If no result found, return None.
        return None

    def matches_template(self, input_text) -> bool:
        r"""Checks if the input_text matches the first unformatted part of the prompt template."""
        index = self.template.find("{")
        return input_text[:index] == self.template[:index]


class ScoringPrompt(BasePrompt):
    def __init__(self):
        super().__init__()
        self.extract_pattern = r"\b([0-9]|10)\b"

    def extract_score(self, response: str) -> float:
        r"""Extract numeric score (range 0-10) from prompt response."""
        # Try to extract score with "Score:" prefix first
        score_match = re.search(r"(?i)score[:\s]*(\d+(?:\.\d+)?)", response)
        if score_match:
            try:
                score = float(score_match.group(1))
                if 0 <= score <= 10:
                    return score
            except ValueError:
                pass

        # Fallback to original extraction
        extraction = self.extract(response)
        if extraction is not None:
            try:
                score = float(extraction)
                if 0 <= score <= 10:
                    return score
            except ValueError:
                return 0
        return 0

    @staticmethod
    def mock_response():
        r"""Mock responses to a followup prompt, for use in MockDendritePool."""
        return random.choices(["", f"Score: {random.randint(0, 10)}"], weights=[1, 9])[
            0
        ]


class SummaryRelevancePrompt(ScoringPrompt):
    """Scores a summary on a strict 4-point scale (0-3) for improved consistency."""

    def __init__(self):
        super().__init__()
        self.template = user_summary_relevance_template
        self.extract_pattern = r"\b([0-3])\b"  # Updated for 0-3 range

    def get_system_message(self) -> str:
        return system_summary_relevance_template

    def extract_score(self, response: str) -> float:
        """Extract numeric score (range 0-3) from prompt response."""
        # Try to extract score with "Score:" prefix first
        score_match = re.search(r"(?i)score[:\s]*([0-3])", response)
        if score_match:
            try:
                score = float(score_match.group(1))
                if 0 <= score <= 3:
                    return score
            except ValueError:
                pass

        # Fallback to original extraction
        extraction = self.extract(response)
        if extraction is not None:
            try:
                score = float(extraction)
                if 0 <= score <= 3:
                    return score
            except ValueError:
                return 0
        return 0


class LinkContentPrompt(ScoringPrompt):
    r"""Scores a summary on a scale from 0 to 10, given a context."""

    def __init__(self):
        super().__init__()
        self.template = user_message_question_answer_template

    def get_system_message(self):
        return system_message_question_answer_template

    def extract_score(self, response: str) -> float:
        r"""Extract numeric score (range 0-10) from prompt response."""
        # Mapping of special codes to numeric scores

        # Extract score from output string with various formats
        match = re.search(r"(?i)score[:\s]*([0-9]|10)", response)
        if match:
            try:
                score = float(match.group(1))
                if 0 <= score <= 10:
                    return score
            except ValueError:
                return 0

        # Extract score directly from the response if "Score:" prefix is missing
        match = re.search(r"\b([0-9]|10)\b", response)
        if match:
            try:
                score = float(match.group(1))
                if 0 <= score <= 10:
                    return score
            except ValueError:
                return 0

        return 0


class SummaryRulePrompt(ScoringPrompt):
    r"""Used to validate if summary was generated following the rules specified"""

    def __init__(self):
        super().__init__()
        self.template = user_summary_validation_template

    def get_system_message(self):
        return system_message_summary_validation_template

    def get_messages(self, summary_text, summary_rule):
        return [
            {
                "role": "system",
                "content": self.get_system_message(),
            },
            {"role": "user", "content": self.text(summary_text, summary_rule)},
        ]

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
        match = re.search(r"\b([0-9]|10)\b", response)
        if match:
            try:
                score = float(match.group(1))
                if 0 <= score <= 10:
                    return score
            except ValueError:
                return 0

        return 0


class LinkContentAndDescriptionPrompt(ScoringPrompt):
    r"""Compares a tweet or link title with summarized description in markdown and prompt
    Used to score each link from twitter or search summary
    """

    def __init__(self):
        super().__init__()
        self.template = text_and_summarized_description_template
        self.weights = {
            "relevance": 0.4,
            "brevity": 0.1,
            "clarity": 0.3,
            "coverage": 0.2,
        }

    def get_system_message(self):
        return text_and_summarized_description_scoring_template

    def extract_score(self, response: str) -> float:
        try:
            scores = json.loads(response)

            final_score = sum(
                scores.get(criterion, 0) * weight / 2
                for criterion, weight in self.weights.items()
            )

            return min(final_score, 1)
        except json.JSONDecodeError:
            return 0


class SearchSummaryRelevancePrompt(ScoringPrompt):
    r"""Scores a summary on a scale from 0 to 10, given a context."""

    def __init__(self):
        super().__init__()
        self.template = user_message_question_answer_template

    def get_system_message(self):
        return system_message_question_answer_template

    def extract_score(self, response: str) -> float:
        r"""Extract numeric score (range 0-10) from prompt response."""
        # Mapping of special codes to numeric scores

        # Extract score from output string with various formats
        match = re.search(r"(?i)score[:\s]*([0-9]|10)", response)
        if match:
            try:
                score = float(match.group(1))
                if 0 <= score <= 10:
                    return score
            except ValueError:
                return 0

        # Extract score directly from the response if "Score:" prefix is missing
        match = re.search(r"\b([0-9]|10)\b", response)
        if match:
            try:
                score = float(match.group(1))
                if 0 <= score <= 10:
                    return score
            except ValueError:
                return 0

        return 0


def find_unique_tags(input_text: str):
    r"""Find all substrings that match the pattern '<...>'."""
    matches = re.findall("<([^>]*)>", input_text)
    # Return a list of unique matches.
    return list(set(matches))


def clean_template(template):
    """Remove leading spaces from each line in the template."""
    # Split the text into lines
    lines = template.split("\n")

    # Remove leading spaces from each line
    cleaned_lines = [line.lstrip() for line in lines]

    # Join the lines back together
    return "\n".join(cleaned_lines)


system_summary_relevance_template = """You are an expert content evaluator assessing AI-generated summaries with strict scoring criteria.

STRUCTURAL REQUIREMENTS (Must ALL be met for any score above 0):
- Headers MUST use ** formatting (e.g., **Header Name**), NOT # or ##
- Summary MUST contain at least 3 markdown links in format [text](url)
- Links must be naturally integrated within the text, not just listed
- Content must directly address the user's query

SCORING CRITERIA:

Score 0 - FAILS BASIC REQUIREMENTS:
- Missing markdown links (fewer than 3) OR
- Uses wrong header format (# or ##) OR
- Completely off-topic or doesn't address the query OR
- Only restates the question without providing substantive content OR
- Empty or extremely brief response

Score 1 - MEETS MINIMUM STANDARDS:
- Has required structural elements (** headers, 3+ links)
- Addresses the query but with shallow or generic content
- Links are present but may not strongly support the claims
- Information is basic and lacks depth or insight

Score 2 - GOOD QUALITY:
- Contains 4+ well-integrated markdown links
- Comprehensively addresses most aspects of the query
- Well-organized with clear, logical structure
- Most claims are properly supported by relevant citations
- Provides useful, actionable information

Score 3 - EXCELLENT QUALITY:
- Contains 5+ highly relevant markdown links
- Fully addresses ALL aspects of the query with depth
- Exceptional organization and professional presentation
- ALL claims backed by strong, relevant citations
- Provides valuable insights and goes beyond basic information
- Demonstrates clear expertise and thorough research

EVALUATION CHECKLIST:
1. Check structural requirements (** headers, 3+ markdown links)
2. Verify content directly addresses the user's query
3. Count and assess quality of markdown links
4. Evaluate depth and comprehensiveness of content
5. Verify citations properly support claims made
6. Assess overall value and insight provided

BE STRICT: When in doubt between two scores, choose the LOWER score. Quality standards must be consistently high.

Output Format:
Score: [0-3]
Explanation: [Specific explanation referencing which criteria were met or failed, including link count and structural assessment]
"""


user_summary_relevance_template = """
<Question>
{}
</Question>

<Answer>
{}
</Answer>
"""


system_message_question_answer_template = """
Relevance Scoring Guide:

Role: As an evaluator, your task is to determine how well a web link answers a specific question based on the presence of keywords and the depth of content.

Scoring Criteria:

Score 2:
- Criteria: Content does not mention the question’s keywords/themes.
- Example:
  - Question: "Effects of global warming on polar bears?"
  - Content: "Visit the best tropical beaches!"
  - Output: Score 2, Explanation: No mention of global warming or polar bears.

Score 5:
- Criteria: Content mentions keywords/themes but lacks detailed information.
- Example:
  - Question: "AI in healthcare?"
  - Content: "AI is transforming industries."
  - Output: Score 5, Explanation: Mentions AI but not healthcare.

Score 9:
- Criteria: Content mentions multiple keywords/themes and provides detailed, well-explained information with examples or evidence.
- Example:
  - Question: "Latest trends in renewable energy?"
  - Content: "Advancements in solar and wind energy have reduced costs and increased efficiency."
  - Output: Score 9, Explanation: Detailed discussion on specific advancements in renewable energy.

Important Rules:
1. Identify Keywords: Extract keywords/themes from the question.
2. Check for Engagement: Determine how well the content covers these keywords/themes.
3. Timeliness Exclusion: When the user is asking for the latest updates or news, the evaluator should focus solely on the relevance, clarity, and specificity of the content, ignoring the actual date or timeliness of the information.
4. Scoring:
   - 2: No relevant keywords.
   - 5: Superficial mention.
   - 9: Detailed, well-explained information with examples or evidence.
   
Output Format:
Score: [2, 5, or 9], Explanation:
"""

system_message_summary_validation_template = """
Scoring Guide

Role: As an evaluator, your task is to evaluate the adherence of a generated summary to specific guidelines outlined in a user system message.
Follow these steps.

1. Input Information
-User System Message: <SummaryRule>
-Generated Summary: <SummaryText>

2. Evaluation Criteria
- Check if the summary captures the main points outlined in the user system message.
- Verify that the tone and style of the summary match the specifications in the user system message
- Assess whether the summary maintains the required length and structure as indicated in the user system message
- Look for any specific keywords or phrases mentioned in the user system message and confirm their presence in the summary.

3. Output
- Assign a score based on adherence:
  - Score 10: If the summary fully adheres to the user system message guidelines
  - Score 0: If the summary does not adhere to the user system message guidelines.

- Output the score and the reason:
  Example output
  - Score 0: The generated summary does not include any of the main points outlined in the user system message and completely deviates from the required tone and style
  - Score 10: The generated summary effectively captures all the main points, mirros the tone and style of the user system message, adheres to the length requirements, and include all the necessary keywords.
"""

text_and_summarized_description_scoring_template = """
# Text and Summary Comparison Mechanism

## 1. Define Criteria for Evaluation
Establish clear criteria for evaluating a summary:
- **Relevance**: Captures the main points of the original text.
- **Brevity**: Concise without losing essential information.
- **Clarity**: Readable and understandable.
- **Coverage**: Comprehensive coverage of key aspects.

## 2. Develop a Scoring Rubric
Create a scoring rubric with specific guidelines for assigning scores. Use a binary scoring system (0 or 1) for each criterion.

### Example Rubric:
- **Relevance**:
  - 2: Captures all main points.
  - 1: Captures some main points but misses others.
  - 0: Misses major points or includes irrelevant details.
- **Brevity**:
  - 2: Concise and to the point.
  - 1: Somewhat concise but could be more succinct.
  - 0: Overly lengthy or too brief.
- **Clarity**:
  - 2: Clear and easy to understand.
  - 1: Some parts are unclear or confusing.
  - 0: Confusing or poorly written.
- **Coverage**:
  - 2: Covers all key aspects.
  - 1: Covers some key aspects but omits others.
  - 0: Omits critical information.

## Output JSON format:
{
  "relevance": 1,
  "brevity": 0,
  "clarity": 2,
  "coverage": 0,
  "explanation": "Explain why each criterion received its score."
}
"""

user_message_question_answer_template = """
Here is the question:
<Question>
{}
</Question>

And the answer content:
<Answer>
{}
</Answer>

Please evaluate the above <Question></Question> and <Answer></Answer> using relevance Scoring Guide in the system message.
"""

text_and_summarized_description_template = """
Here is the text content:
<Text>
{}
</Text>

And the summarized description of the text content:
<SummarizedText>
{}
</SummarizedText>

Please evaluate the above, <Text></Text> and <SummarizedText></SummarizedText> using relevance Scoring Guide in the system message.
"""

user_summary_validation_template = """
Here is summarized text:
<SummaryText>
{}
</SummaryText>

And the rules for generating summary:
<SummaryRule>
{}
</SummaryRule>

Please evaluate the above, <SummaryText></SummaryText> and <SummaryRule></SummaryRule> using relevance Guide in the system message.
"""
