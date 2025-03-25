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
import time
import torch
import bittensor as bt
import random
import asyncio
import re
from typing import List, Tuple
from neurons.validators.reward.config import RewardModelType, RewardScoringType
from neurons.validators.reward.reward import BaseRewardModel, BaseRewardEvent
from neurons.validators.utils.prompts import (
    SummaryRelevancePrompt,
    LinkContentPrompt,
    LinkContentAndDescriptionPrompt,
)

from datura.protocol import ScraperStreamingSynapse, ScraperTextRole
from neurons.validators.reward.reward_llm import RewardLLM
from datura.services.twitter_utils import TwitterUtils
from datura.services.web_search_utils import WebSearchUtils
import json
from neurons.validators.reward.config import DefaultSummaryRelevanceWeightConfig
from datura.utils import clean_text
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

    def get_scoring_text(
        self, response: ScraperStreamingSynapse, random_completion: Tuple[str, str]
    ) -> BaseRewardEvent:
        try:
            # Score random completion
            summary_key, completion = random_completion

            if not completion:
                return None

            is_final_summary = (
                response.result_type == ResultType.LINKS_WITH_FINAL_SUMMARY
                and summary_key == ScraperTextRole.FINAL_SUMMARY.value
            )

            completion = self.validate_successful_completion(
                response=response, completion=completion
            )

            if not completion:
                return None

            if not self.scoring_type:
                return None

            scoring_prompt = None
            scoring_prompt_text = None

            if is_final_summary:
                # Combine all fields dynamically
                scoring_prompt = LinkContentAndDescriptionPrompt()

                # Get completions based on tools used
                completions, links_expected = response.get_search_completion()

                # Fetch the final summary from synapse.texts if available
                final_summary = response.texts.get("summary", "").strip()
                if not final_summary:
                    return None
                
                # Combine the 'completions' dictionary directly with the final summary
                structured_data = {
                    **completions,
                    "summary": final_summary
                }

                scoring_prompt_text = scoring_prompt.text(response.prompt, structured_data)
            elif self.scoring_type.value == RewardScoringType.summary_relevance_score_template.value:
                scoring_prompt = SummaryRelevancePrompt()
            elif self.scoring_type.value == RewardScoringType.link_content_relevance_template.value:
                scoring_prompt = LinkContentPrompt()
                completion_links_str = str(response.completion_links)
                scoring_prompt_text = scoring_prompt.text(completion, completion_links_str)

            # Check completion links based on summary type
            has_required_links = True
            if summary_key == ScraperTextRole.TWITTER_SUMMARY.value:
                has_required_links = bool(response.completion_links)
            else:
                has_required_links = bool(response.search_completion_links)

            if scoring_prompt is None or not has_required_links:
                return None

            if not scoring_prompt_text:
                scoring_prompt_text = scoring_prompt.text(response.prompt, completion)

            return scoring_prompt, [
                {
                    "role": "system",
                    "content": scoring_prompt.get_system_message(
                        tools=response.tools,
                        result_type=response.result_type,
                        summary_key=summary_key
                    ),
                },
                {"role": "user", "content": scoring_prompt_text},
            ]

        except Exception as e:
            bt.logging.error(f"Summary Relevance get_scoring_text: {str(e)}")
            return None



    async def process_link_scoring_messages(
        self,
        responses: List[ScraperStreamingSynapse],
        random_completions: List[Tuple[str, str]],
    ):
        scoring_messages = {}

        scoring_prompt = LinkContentAndDescriptionPrompt()

        scoring_keys_list = []

        # Accumulate scoring messages to compare scraped tweet texts with markdown link descriptions
        for response, random_completion in zip(responses, random_completions):
            summary_key, completion = random_completion

            if not completion:
                scoring_keys_list.append([])
                continue

            is_twitter = summary_key == ScraperTextRole.TWITTER_SUMMARY.value
            link_with_descriptions = []

            # Parse markdown links with descriptions from completion
            if is_twitter:
                link_with_descriptions = (
                    TwitterUtils().find_twitter_link_with_descriptions(completion or "")
                )
            else:
                link_with_descriptions = WebSearchUtils.find_links_with_descriptions(
                    completion or ""
                )

            scoring_keys = []

            for link, description in link_with_descriptions:
                link = WebSearchUtils.remove_trailing_slash(link)
                text = ""
                scoring_key = ""

                # Find validator scraped tweet or link to compare with miner's link description
                if is_twitter:
                    validator_tweet = next(
                        (
                            validator_tweet
                            for validator_tweet in response.validator_tweets
                            if f"{validator_tweet.user.username}/status/{validator_tweet.id}"
                            in link
                        ),
                        None,
                    )

                    if not validator_tweet:
                        continue

                    text = clean_text(validator_tweet.text)
                else:
                    validator_link = next(
                        (
                            validator_link
                            for validator_link in response.validator_links
                            if WebSearchUtils.remove_trailing_slash(
                                validator_link.get("url", "")
                            )
                            == link
                        ),
                        None,
                    )

                    if not validator_link:
                        continue

                    text = validator_link.get("title")

                scoring_key = f"{link}/{description}"
                scoring_prompt_text = scoring_prompt.text(text, description)

                scoring_text = [
                    {
                        "role": "system",
                        "content": scoring_prompt.get_system_message(),
                    },
                    {"role": "user", "content": scoring_prompt_text},
                ]

                scoring_keys.append(scoring_key)
                scoring_messages[scoring_key] = scoring_text

            scoring_keys_list.append(scoring_keys)

        scoring_messages = [
            {scoring_key: scoring_text}
            for scoring_key, scoring_text in scoring_messages.items()
        ]

        if not scoring_messages:
            return [0 for _ in responses], scoring_keys_list

        # Process scoring messages in groups to avoid the OpenAI timeouts
        group_size = 200

        scoring_messages_groups = [
            scoring_messages[i : i + group_size]
            for i in range(0, len(scoring_messages), group_size)
        ]

        score_responses = {}

        for scoring_messages_group in scoring_messages_groups:
            score_responses_group = await self.reward_llm.llm_processing(
                scoring_messages_group
            )
            score_responses.update(score_responses_group)

        return score_responses, scoring_keys_list

    async def score_link_descriptions(
        self,
        responses: List[ScraperStreamingSynapse],
        uids,
        random_completions: List[Tuple[str, str]],
    ):
        score_responses, scoring_keys_list = await self.process_link_scoring_messages(
            responses, random_completions
        )

        scoring_prompt = LinkContentAndDescriptionPrompt()

        average_scores = []
        link_description_scores_list = []

        expected_links = 10

        # Scoring keys list maintains the order of responses
        for scoring_keys in scoring_keys_list:
            # Store link scores and link with their scores of each response
            link_description_scores = []
            link_description_scores_map = {}

            # Parse scores from LLM response for each link
            for scoring_key in scoring_keys:
                score_result = score_responses.get(scoring_key)
                score = scoring_prompt.extract_score(score_result)

                if score is not None:
                    link_description_scores_map[scoring_key] = score
                    link_description_scores.append(score)

            # Calculate average score and scale down to 0-1 range
            average_score = sum(link_description_scores) / expected_links
            average_score = min(average_score, 1)
            average_scores.append(average_score)
            link_description_scores_list.append(link_description_scores_map)

        uid_to_average_score = dict(zip(uids.tolist(), average_scores))

        bt.logging.info(
            f"SummaryRelevanceRewardModel | average scores: {uid_to_average_score}"
        )

        return average_scores, link_description_scores_list

    async def get_rewards(
        self, responses: List[ScraperStreamingSynapse], uids
    ) -> List[BaseRewardEvent]:
        try:
            bt.logging.debug(
                f"SummaryRelevanceRewardModel | Calculating {len(responses)} rewards (typically < 1 sec/reward)."
            )

            # Same random completion will be used for all responses
            random_summary_key = random.choice(
                list(responses[0].get_all_completions().keys())
            )

            # Choose random toolkit summary to score
            random_completions = []

            for response in responses:
                # Get all available completions based on the tools used and choose random summary (Twitter, Search, Reddit, Hacker News)
                completions = response.get_all_completions()
                random_completion = completions.get(random_summary_key, "")

                # Check if all summaries are returned
                is_full = all(completion for completion in completions.values())

                random_completions.append(
                    (random_summary_key, random_completion)
                    if random_completion and is_full
                    else (None, None)
                )

            # Need to use this scores to calculate rewards
            (
                average_link_scores,
                link_description_scores_list,
            ) = await self.score_link_descriptions(responses, uids, random_completions)

            scoring_messages = [
                self.get_scoring_text(response, random_completion)
                for response, random_completion in zip(responses, random_completions)
            ]
            filter_scoring_messages = [
                msg for msg in scoring_messages if msg is not None
            ]
            bt.logging.debug(
                f"SummaryRelevanceRewardModel | Calculating {len(filter_scoring_messages)} rewards (typically < 1 sec/reward)."
            )

            # # Filter out None items from scoring_messages
            # messages = []
            # messages.extend({index: msg_content} for index, (_, msg_content) in enumerate(scoring_messages) if msg_content)
            # messages = [{str(index): msg_content} for index, (_, msg_content) in enumerate(filter_scoring_messages)]
            messages = [
                {str(index): item[1]}
                for index, item in enumerate(scoring_messages)
                if item is not None
            ]

            scores = {}
            score_text = {}
            if messages:
                bt.logging.info(
                    f"Executing llm_processing on {len(messages)} summary relevance messages."
                )
                score_responses = await self.reward_llm.llm_processing(messages)

                if score_responses and isinstance(
                    score_responses, dict
                ):  # Ensure score_responses is a dictionary
                    for (key, score_result), (scoring_prompt, _) in zip(
                        score_responses.items(), filter_scoring_messages
                    ):
                        if (
                            score_result is not None
                        ):  # Check if score_result is not None
                            score = scoring_prompt.extract_score(score_result)
                            # Scale 0-10 score to 0-1 range.
                            score /= 10.0
                            scores[key] = score
                            score_text[key] = score_result

            # Iterate over responses and assign rewards based on scores
            reward_events = []

            # Initialize dictionaries to store zero and non-zero scores separately
            zero_scores = {}
            non_zero_scores = {}

            summary_weight = torch.tensor(
                DefaultSummaryRelevanceWeightConfig.summary_weight,
                device=self.device,
                dtype=torch.float32,
            )

            link_content_weight = torch.tensor(
                DefaultSummaryRelevanceWeightConfig.link_content_weight,
                device=self.device,
                dtype=torch.float32,
            )

            for (index, response), average_link_score, uid_tensor in zip(
                enumerate(responses), average_link_scores, uids
            ):
                uid = uid_tensor.item()

                if response.result_type == ResultType.ONLY_LINKS:
                    reward_event = BaseRewardEvent()
                    reward_event.reward = 1.0
                    reward_events.append(reward_event)
                    continue

                summary_score = scores.get(str(index), 0)

                summary_score = (
                    torch.tensor(summary_score, device=self.device, dtype=torch.float32)
                    * summary_weight
                )

                links_score = (
                    torch.tensor(
                        average_link_score, device=self.device, dtype=torch.float32
                    )
                    * link_content_weight
                )

                score = None

                # If whole summary content is failed, ignore link scores
                if summary_score != 0:
                    score = torch.clamp(summary_score + links_score, max=1.0)
                else:
                    score = torch.tensor(0, device=self.device, dtype=torch.float32)

                score = score.item()
                score_explain = score_text.get(str(index), "")

                reward_event = BaseRewardEvent()
                reward_event.reward = score
                reward_events.append(reward_event)

                if score == 0:
                    zero_scores[uid] = score
                else:
                    non_zero_scores[uid] = score

            bt.logging.info(
                f"==================================Summary Relevance scoring Zero Scores  ({len(zero_scores)} cases)=================================="
            )
            bt.logging.info(json.dumps(zero_scores))
            bt.logging.info(
                f"==================================Summary Relevance scoring Non-Zero Scores ({len(non_zero_scores)} cases)=================================="
            )
            bt.logging.info(json.dumps(non_zero_scores))

            return reward_events, link_description_scores_list
        except Exception as e:
            error_message = f"Summary Relevance get_rewards: {str(e)}"
            tb_str = traceback.format_exception(type(e), e, e.__traceback__)
            bt.logging.error("\n".join(tb_str) + error_message)
            reward_events = []
            for response in responses:
                reward_event = BaseRewardEvent()
                reward_event.reward = 0
                reward_events.append(reward_event)
            return reward_events, []
