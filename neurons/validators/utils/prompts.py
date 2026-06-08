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

import random
import re


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


_LABEL_TO_SCORE = {"HIGH": 3.0, "MEDIUM": 2.0, "LOW": 1.0, "OFFTOPIC": 0.0}
_LABEL_RE = re.compile(r"(?i)\b(HIGH|MEDIUM|LOW|OFFTOPIC)\b")


def _extract_label_score(response: str) -> float:
    if not response:
        return 0.0
    if m := _LABEL_RE.search(response):
        return _LABEL_TO_SCORE[m.group(1).upper()]
    if m := re.search(r"(?i)score\s*[:\s]+([0-3])\b", response):
        return float(m.group(1))
    return 0.0


class SummaryRelevancePrompt(ScoringPrompt):
    """Scores a summary on a 4-tier label scale (OFFTOPIC / LOW / MEDIUM / HIGH)."""

    def __init__(self):
        super().__init__()
        self.template = user_summary_relevance_template
        self.extract_pattern = r"(?i)\b(HIGH|MEDIUM|LOW|OFFTOPIC)\b"

    def get_system_message(self) -> str:
        return system_summary_relevance_template

    def extract_score(self, response: str) -> float:
        return _extract_label_score(response)


class LinkContentPrompt(ScoringPrompt):
    r"""Scores a link/tweet on a 4-tier label scale (OFFTOPIC / LOW / MEDIUM / HIGH)."""

    def __init__(self):
        super().__init__()
        self.template = user_message_question_answer_template
        self.extract_pattern = r"(?i)\b(HIGH|MEDIUM|LOW|OFFTOPIC)\b"

    def get_system_message(self):
        return system_message_question_answer_template

    def extract_score(self, response: str) -> float:
        return _extract_label_score(response)


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


class SearchSummaryRelevancePrompt(ScoringPrompt):
    r"""Scores a search-result link on a 4-tier label scale (OFFTOPIC / LOW / MEDIUM / HIGH)."""

    def __init__(self):
        super().__init__()
        self.template = user_message_question_answer_template
        self.extract_pattern = r"(?i)\b(HIGH|MEDIUM|LOW|OFFTOPIC)\b"

    def get_system_message(self):
        return system_message_question_answer_template

    def extract_score(self, response: str) -> float:
        return _extract_label_score(response)


system_message_web_search_relevance_template = """You judge whether a web page is a relevant result for a search query — the way a normal search engine would. You see only the page's title and description (snippet), not the full page.

RELEVANT — the page is genuinely about the query's topic: what someone running this search would reasonably expect to find. How-to pages, documentation, articles, news, forum threads, videos, overviews, and listing / hub pages all count as RELEVANT when they are on the query's topic.

IRRELEVANT — the page is about a different topic and only matches the query through a shared common word. Example: the query is about "docker" but the result is a video titled "What hard work looks like" that matched only on the word "work". If the page's topic does not match the query's topic, it is IRRELEVANT.

Be lenient: a search engine legitimately returns a wide range of on-topic results. Only answer IRRELEVANT when the topic genuinely does not match the query.

Output exactly one word: RELEVANT or IRRELEVANT."""


user_message_web_search_relevance_template = """
<Query>
{}
</Query>

<Result>
{}
</Result>
"""


class WebSearchRelevancePrompt(ScoringPrompt):
    r"""Binary relevance for a plain web-search result (RELEVANT / IRRELEVANT) — topic match, not answer-bearing intent."""

    def __init__(self):
        super().__init__()
        self.template = user_message_web_search_relevance_template
        self.extract_pattern = r"(?i)\b(IRRELEVANT|RELEVANT)\b"

    def get_system_message(self):
        return system_message_web_search_relevance_template

    def extract_score(self, response: str) -> float:
        if not response:
            return 0.0
        text = response.upper()
        if "IRRELEVANT" in text or "NOT RELEVANT" in text:
            return 0.0
        if "RELEVANT" in text:
            return 1.0
        return 0.0


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


system_summary_relevance_template = """You judge whether an AI-generated answer SUMMARY satisfies the user's query.

Two things matter independently — BOTH gate the verdict:
  (A) Does the answer deliver the requested fact?
  (B) Is the answer's evidence trail honest — every specific factual claim cited to a source that plausibly contains that claim?

Pick exactly ONE verdict:

HIGH — Fully addresses every part of the question with appropriate depth AND has tight, distributed citations: each specific factual claim is backed by a source whose URL or title shows it is the answer-bearing page for that claim. Distinct claims do not all stack on a single source. Where authoritative sources disagree, the answer notes the variation and cites each. Honest "data not yet available" / "event has not happened" answers that show the right sources were checked qualify here too.

MEDIUM — Directly answers the question, and the citation trail is plausible: claims are cited to URLs whose titles or paths suggest they contain the specific fact. Synthesis or framing sentences can go uncited; specific numbers, names, and dates should be cited to specific pages.

LOW — Either the user's specific question is not actually answered (it dances around the topic), OR the fact is delivered but the citation trail is broken (cited URLs are clearly the wrong page for the claim, or the same source is stacked across many unrelated claims).

OFFTOPIC — Empty, dodges the question, restates it without answering, or contains contradictions / clear hallucinations. Also OFFTOPIC if most factual claims have no citation at all.

Principles:
- RIGHT-SIZED beats verbose. A short focused answer to a simple question can be HIGH. Do not reward length.
- Ranges count as answers. "$77,300–$77,400 across exchanges" answers a price question.
- A citation is only credible if the cited page's title or URL path suggests it contains the specific claim. Hub / calendar / index pages do not back specific facts.
- When uncertain between two verdicts, pick the LOWER one.

Output EXACTLY two lines, nothing else:
Verdict: <HIGH|MEDIUM|LOW|OFFTOPIC>
Reason: <one short sentence, max 25 words>
"""


user_summary_relevance_template = """
<Question>
{}
</Question>

<Answer>
{}
</Answer>
"""


system_message_question_answer_template = """You judge whether a SOURCE (web page or tweet) is useful for answering a user query — by INTENT match, not keyword overlap. The input is metadata (title + description / snippet), NOT the full page content. Judge how likely THIS specific page contains the answer the user needs.

DIRECT-ANSWER TEST (apply first): does the title or snippet contain the specific evidence a user would need to answer THIS question? "What X says" needs quotes from X (not discussion ABOUT X). "Current X / latest X" needs the value or name. "How to X" needs steps or mechanism. "Who won / what happened" needs the outcome. If yes → HIGH. If no → LOW or lower.

Pick exactly ONE verdict:

HIGH — Answer-bearing page. The user could open this single tab and have what they need.

MEDIUM — On-topic but a hub / index / category page (calendars, year-indexes, "list of" pages, topic landings). Also MEDIUM when the entity / topic matches but the subtopic is narrower or broader than the query, or the snippet hints without confirming.

LOW — In the same broad domain but does not match the user's specific intent: wrong subtopic, wrong entity, or meta-discussion of the topic without the direct evidence the question asks for.

OFFTOPIC — Different topic entirely. Neither title nor snippet has a semantic link to the query's intent.

Principles:
- Judge by INTENT against the specific question. A topic match alone is not enough.
- TITLE and SNIPPET are BOTH strong signals. A generic-looking title is still HIGH if the snippet contains the specific answer.
- Hub markers in the TITLE ("list of", "calendar", "index", "all X", "guide to") lean MEDIUM — UNLESS the snippet quotes the specific answer, then HIGH.
- Do NOT penalize stale content. Date filtering happens elsewhere.
- When uncertain between two verdicts, pick the LOWER one.

Output EXACTLY two lines, nothing else:
Verdict: <HIGH|MEDIUM|LOW|OFFTOPIC>
Reason: <one short sentence, max 20 words>
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
