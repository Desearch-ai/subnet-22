# The MIT License (MIT)
# Copyright © 2023 Yuma Rao
# Copyright © 2023 Opentensor Foundation

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.p
import traceback
import bittensor as bt
import asyncio
import re
from typing import List, Tuple, Dict
from neurons.validators.reward.config import RewardModelType
from neurons.validators.reward.reward import BaseRewardModel, BaseRewardEvent
from neurons.validators.utils.prompts import SummaryRelevancePrompt
from neurons.validators.reward.reward_llm import RewardLLM
from datura.protocol import ScraperStreamingSynapse, ScraperTextRole
from datura.services.web_search_utils import WebSearchUtils
from datura.protocol import ResultType


class SummaryRelevanceRewardModel(BaseRewardModel):
    reward_model_name: str = "VMware/open-llama-7b-open-instruct"

    @property
    def name(self) -> str:
        return RewardModelType.summary_relavance_match.value

    def __init__(self, device: str, scoring_type: None, llm_reward: RewardLLM):
        super().__init__()
        self.device = device
        self.reward_llm = llm_reward
        self.scoring_type = scoring_type

    def extract_links_from_markdown(self, text: str) -> List[Tuple[str, str]]:
        """Extract markdown links with their text from the completion.
        Returns list of (link_text, url) tuples."""
        pattern = r"\[([^\]]+)\]\(([^)]+)\)"
        matches = re.findall(pattern, text)
        return matches

    def check_markdown_structure(self, text: str) -> Tuple[bool, List[str]]:
        """Check if markdown follows proper structure.
        Returns (is_valid, list_of_issues)"""
        issues = []

        # Check for improper header usage
        if re.search(r"^#{1,6}\s", text, re.MULTILINE):
            issues.append("Uses # headers instead of **")

        # Check if headers exist using **
        header_pattern = r"\*\*[^*]+\*\*"
        headers = re.findall(header_pattern, text)
        if len(headers) < 1:
            issues.append("No proper headers found (should use ** for headers)")

        # Check for basic structure
        if not text.strip():
            issues.append("Empty response")

        return len(issues) == 0, issues

    def verify_link_sources(
        self, response: ScraperStreamingSynapse, links: List[str]
    ) -> Tuple[int, int, Dict[str, bool]]:
        """Verify that links in the markdown exist in the miner's sources.
        Returns (verified_count, total_count, link_verification_map)"""
        verified_count = 0
        link_verification = {}

        # Normalize links for comparison
        def normalize_url(url: str) -> str:
            url = url.strip()
            # Remove trailing slashes
            url = WebSearchUtils.remove_trailing_slash(url)
            # Handle twitter/x.com domains
            url = url.replace("https://twitter.com/", "https://x.com/")
            return url.lower()

        # Collect all valid source links
        valid_sources = set()

        # Add Twitter sources
        if response.miner_tweets:
            for tweet in response.miner_tweets:
                if isinstance(tweet, dict):
                    username = tweet.get("user", {}).get("username", "")
                    tweet_id = tweet.get("id", "")
                    if username and tweet_id:
                        twitter_url = f"https://x.com/{username}/status/{tweet_id}"
                        valid_sources.add(normalize_url(twitter_url))

        # Add search results from all search tools
        search_results_fields = [
            "search_results",
            "wikipedia_search_results",
            "youtube_search_results",
            "arxiv_search_results",
            "reddit_search_results",
            "hacker_news_search_results",
        ]

        for field in search_results_fields:
            results = getattr(response, field, [])
            if results:
                for result in results:
                    if isinstance(result, dict) and "link" in result:
                        valid_sources.add(normalize_url(result["link"]))
                    elif hasattr(result, "link") and result.link:
                        valid_sources.add(normalize_url(result.link))

        # Verify each link
        for link in links:
            normalized_link = normalize_url(link)
            is_verified = normalized_link in valid_sources
            link_verification[link] = is_verified
            if is_verified:
                verified_count += 1

        return verified_count, len(links), link_verification

    async def score_final_summary(
        self, response: ScraperStreamingSynapse
    ) -> Tuple[float, str, Dict]:
        """Score the final summary with detailed evaluation."""
        try:
            # Get the final summary
            final_summary = response.texts.get(ScraperTextRole.FINAL_SUMMARY.value, "")

            if not final_summary:
                return 0.0, "No final summary found", {}

            # Check markdown structure
            is_valid_structure, structure_issues = self.check_markdown_structure(
                final_summary
            )

            # Extract links
            link_matches = self.extract_links_from_markdown(final_summary)
            links = [url for _, url in link_matches]

            if not links:
                return (
                    0.0,
                    "No links found in summary",
                    {
                        "structure_valid": is_valid_structure,
                        "structure_issues": structure_issues,
                        "link_count": 0,
                    },
                )

            # Verify link sources
            verified_count, total_count, link_verification = self.verify_link_sources(
                response, links
            )

            # If less than 50% of links are verified, heavily penalize
            verification_ratio = verified_count / total_count if total_count > 0 else 0

            if verification_ratio < 0.5:
                return (
                    0.0,
                    f"Insufficient link verification: {verified_count}/{total_count} links verified",
                    {
                        "structure_valid": is_valid_structure,
                        "structure_issues": structure_issues,
                        "link_count": total_count,
                        "verified_links": verified_count,
                        "link_verification": link_verification,
                    },
                )

            # Get LLM scoring using SummaryRelevancePrompt
            scoring_prompt = SummaryRelevancePrompt()

            scoring_messages = [
                {
                    "0": [
                        {
                            "role": "system",
                            "content": scoring_prompt.get_system_message(),
                        },
                        {
                            "role": "user",
                            "content": scoring_prompt.text(
                                response.prompt, final_summary
                            ),
                        },
                    ]
                }
            ]

            score_responses = await self.reward_llm.llm_processing(scoring_messages)

            llm_score = 0.0
            score_explanation = "Failed to get LLM score"

            if score_responses and "0" in score_responses:
                score_text = score_responses["0"]
                llm_score = scoring_prompt.extract_score(score_text)
                llm_score = llm_score / 3.0  # Normalize 0-3 to 0-1
                score_explanation = score_text

            # Apply penalties for structure issues
            structure_penalty = 0.1 if not is_valid_structure else 0

            # Apply bonus for link verification ratio
            verification_bonus = 0.1 * verification_ratio

            # Calculate final score
            final_score = max(
                0, min(1, llm_score - structure_penalty + verification_bonus)
            )

            # Apply hard penalty if verification is too low
            if verification_ratio < 0.7:
                final_score *= verification_ratio

            return (
                final_score,
                score_explanation,
                {
                    "structure_valid": is_valid_structure,
                    "structure_issues": structure_issues,
                    "link_count": total_count,
                    "verified_links": verified_count,
                    "verification_ratio": verification_ratio,
                    "link_verification": link_verification,
                    "llm_score": llm_score,
                    "penalties_applied": {
                        "structure_penalty": structure_penalty,
                        "verification_bonus": verification_bonus,
                    },
                },
            )

        except Exception as e:
            bt.logging.error(f"Error in score_final_summary: {str(e)}")
            return 0.0, str(e), {}

    async def get_rewards(
        self, responses: List[ScraperStreamingSynapse], uids
    ) -> Tuple[List[BaseRewardEvent], List[Dict]]:
        """Calculate rewards for responses based on new scoring mechanism."""
        try:
            bt.logging.debug(
                f"SummaryRelevanceRewardModel | Calculating {len(responses)} rewards."
            )

            reward_events = []
            scoring_details = []

            # Process responses in batches to avoid timeouts
            batch_size = 50

            for i in range(0, len(responses), batch_size):
                batch_responses = responses[i : i + batch_size]
                batch_uids = uids[i : i + batch_size]

                # Score each response in the batch
                batch_tasks = []
                for response in batch_responses:
                    if response.result_type == ResultType.LINKS_WITH_FINAL_SUMMARY:
                        batch_tasks.append(self.score_final_summary(response))
                    else:
                        # For non-final summary types, give default score
                        batch_tasks.append(self._default_score(response))

                # Wait for all scores in batch
                batch_results = await asyncio.gather(*batch_tasks)

                # Create reward events
                for (score, explanation, details), response, uid in zip(
                    batch_results, batch_responses, batch_uids
                ):
                    reward_event = BaseRewardEvent(reward=score)
                    reward_events.append(reward_event)

                    scoring_details.append(
                        {
                            "uid": uid.item() if hasattr(uid, "item") else uid,
                            "score": score,
                            "explanation": explanation,
                            "details": details,
                        }
                    )

            # Log scoring results
            zero_scores = {
                d["uid"]: d["score"] for d in scoring_details if d["score"] == 0
            }
            non_zero_scores = {
                d["uid"]: d["score"] for d in scoring_details if d["score"] > 0
            }

            bt.logging.info(
                f"{'='*30} Summary Relevance Zero Scores ({len(zero_scores)} cases) {'='*30}"
            )
            if zero_scores:
                bt.logging.info(zero_scores)

            bt.logging.info(
                f"{'='*30} Summary Relevance Non-Zero Scores ({len(non_zero_scores)} cases) {'='*30}"
            )
            if non_zero_scores:
                bt.logging.info(non_zero_scores)

            return reward_events, scoring_details

        except Exception as e:
            error_message = f"Summary Relevance get_rewards error: {str(e)}"
            tb_str = traceback.format_exception(type(e), e, e.__traceback__)
            bt.logging.error("\n".join(tb_str) + error_message)

            # Return zero rewards on error
            reward_events = [BaseRewardEvent(reward=0) for _ in responses]
            return reward_events, []

    async def _default_score(
        self, response: ScraperStreamingSynapse
    ) -> Tuple[float, str, Dict]:
        """Default scoring for non-final summary response types."""
        if response.result_type == ResultType.ONLY_LINKS:
            # For ONLY_LINKS type, check if links are present
            links, _ = response.get_links_from_search_results()
            if links:
                return (
                    1.0,
                    "ONLY_LINKS type with valid links",
                    {"link_count": len(links)},
                )
            else:
                return 0.0, "ONLY_LINKS type but no links found", {"link_count": 0}
        else:
            # For other types, give base score
            return 1.0, f"Response type {response.result_type} - default score", {}
