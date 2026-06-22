import asyncio
import re
import traceback
from typing import Dict, List, Tuple

import bittensor as bt

from desearch.protocol import ResultType, ScraperStreamingSynapse, ScraperTextRole
from neurons.validators.base_validator import AbstractNeuron
from neurons.validators.reward.config import RewardModelType
from neurons.validators.reward.reward import (
    BaseRewardEvent,
    BaseRewardModel,
    log_reward_aggregates,
)
from neurons.validators.reward.reward_llm import RewardLLM
from neurons.validators.utils.prompts import (
    SummaryGroundednessPrompt,
    render_cited_sources,
)
from neurons.validators.utils.response_checks import extract_markdown_links
from neurons.validators.utils.source_bodies import (
    align_citation_markers,
    collect_cited_bodies,
)


class SummaryRelevanceRewardModel(BaseRewardModel):
    reward_model_name: str = "VMware/open-llama-7b-open-instruct"

    @property
    def name(self) -> str:
        return RewardModelType.summary_relavance_match.value

    def __init__(
        self,
        scoring_type: None,
        llm_reward: RewardLLM,
        neuron: AbstractNeuron,
    ):
        super().__init__(neuron)
        self.reward_llm = llm_reward
        self.scoring_type = scoring_type

    def _cited_source_urls(self, response: ScraperStreamingSynapse) -> List[str]:
        summary = response.texts.get(ScraperTextRole.FINAL_SUMMARY.value, "")
        cited = list(dict.fromkeys(url for _, url in extract_markdown_links(summary)))
        if not cited:
            search_links, _ = response.get_links_from_search_results()
            cited = list(dict.fromkeys(search_links))
        return cited

    async def score_final_summary(
        self, response: ScraperStreamingSynapse
    ) -> Tuple[float, str, Dict]:
        try:
            final_summary = response.texts.get(ScraperTextRole.FINAL_SUMMARY.value, "")
            if not final_summary:
                return 0.0, "No final summary found", {}

            cited = self._cited_source_urls(response)
            if not cited:
                return 0.0, "No cited sources to ground against", {}

            bodies = collect_cited_bodies(response, cited)
            if not bodies:
                return (
                    0.0,
                    "No cited source body available",
                    {"cited": cited, "grounded": 0},
                )

            scoring_prompt = SummaryGroundednessPrompt()
            grounded_summary = align_citation_markers(final_summary, bodies)
            user_content = scoring_prompt.text(
                response.prompt, grounded_summary, render_cited_sources(bodies)
            )
            scoring_messages = [
                {
                    "0": [
                        {
                            "role": "system",
                            "content": scoring_prompt.get_system_message(),
                        },
                        {"role": "user", "content": user_content},
                    ]
                }
            ]

            score_responses = await self.reward_llm.llm_processing(scoring_messages)
            if not score_responses or "0" not in score_responses:
                return 0.0, "Failed to get LLM score", {}

            score_text = score_responses["0"]
            if score_text and not re.search(scoring_prompt.extract_pattern, score_text):
                bt.logging.warning(
                    f"Groundedness judge returned no parseable verdict: {score_text[:160]!r}"
                )
            llm_score = scoring_prompt.extract_score(score_text) / 3.0

            return (
                max(0.0, min(1.0, llm_score)),
                score_text,
                {
                    "llm_score": llm_score,
                    "cited": cited,
                    "grounded": len(bodies),
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

            log_reward_aggregates(
                name=self.name,
                uids=[d["uid"] for d in scoring_details],
                scores=[d["score"] for d in scoring_details],
            )

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
            search_links, _ = response.get_links_from_search_results()
            tweet_links = response.get_links_from_tweets()
            links = search_links + tweet_links

            if links:
                if response.completion or response.text_chunks:
                    return 0.0, "ONLY_LINKS type with summary", {}

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
