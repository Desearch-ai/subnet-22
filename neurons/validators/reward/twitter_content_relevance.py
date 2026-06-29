import re
import time
import traceback
from datetime import datetime
from typing import List

import bittensor as bt
import pytz

from desearch.protocol import (
    ScraperStreamingSynapse,
    ScraperTextRole,
    TwitterScraperTweet,
)
from desearch.services.twitter_api_wrapper import TwitterAPIClient
from desearch.services.twitter_utils import TwitterUtils
from desearch.utils import (
    clean_text,
    format_text_for_match,
    is_valid_tweet,
    scrape_tweets_with_retries,
)
from neurons.validators.base_validator import AbstractNeuron
from neurons.validators.penalty.count_penalty import TWITTER_TOOL
from neurons.validators.reward.reward_llm import RewardLLM
from neurons.validators.utils.prompts import (
    TweetRelevancePrompt,
    build_tweet_relevance_messages,
)
from neurons.validators.utils.source_bodies import (
    cited_urls_normalized,
    sample_cited_and_uncited,
    tweet_relevance_text,
)

from .config import RewardModelType
from .reward import (
    BaseRewardEvent,
    BaseRewardModel,
    log_reward_aggregates,
    pattern_to_check,
)

APIFY_LINK_SCRAPE_AMOUNT = 3
MAX_CITED_SAMPLE = 2


def response_uses_twitter_tool(response: ScraperStreamingSynapse) -> bool:
    return TWITTER_TOOL in set(response.tools or [])


class TwitterContentRelevanceModel(BaseRewardModel):
    reward_model_name: str = "VMware/open-llama-7b-open-instruct"

    @property
    def name(self) -> str:
        return RewardModelType.twitter_content_relevance.value

    def __init__(
        self,
        scoring_type: None,
        llm_reward: RewardLLM,
        neuron: AbstractNeuron,
    ):
        super().__init__(neuron)
        self.reward_llm = llm_reward

        self.scoring_type = scoring_type
        self.tw_client = TwitterAPIClient()

    def clean_text(self, text):
        return clean_text(text)

    @staticmethod
    def build_relevance_content(validator_tweet: TwitterScraperTweet) -> str:
        return tweet_relevance_text(validator_tweet)

    async def llm_process_validator_tweets(self, response: ScraperStreamingSynapse):
        if not response_uses_twitter_tool(response) or not response.validator_tweets:
            return {}, 0.0

        start_llm_time = time.time()
        scoring_messages = []
        for validator_tweet in response.validator_tweets:
            body = self.build_relevance_content(validator_tweet)
            author = getattr(getattr(validator_tweet, "user", None), "username", None)
            title = f"Tweet by @{author}" if author else "Tweet"
            url = getattr(validator_tweet, "url", "") or ""
            val_tweet_id = validator_tweet.id
            messages = build_tweet_relevance_messages(response.prompt, url, title, body)
            if messages:
                scoring_messages.append({str(val_tweet_id): messages})
        score_responses = await self.reward_llm.llm_processing(scoring_messages)

        return score_responses, time.time() - start_llm_time

    @staticmethod
    def _sample_cited_and_other_tweets(response) -> List[str]:
        summary = response.texts.get(ScraperTextRole.FINAL_SUMMARY.value, "")
        cited_norm = cited_urls_normalized(summary)
        urls = [
            tweet.get("url")
            for tweet in response.miner_tweets
            if tweet.get("url") and TwitterUtils.is_valid_twitter_link(tweet.get("url"))
        ]
        return sample_cited_and_uncited(
            urls, cited_norm, MAX_CITED_SAMPLE, APIFY_LINK_SCRAPE_AMOUNT
        )

    async def process_tweets(self, responses: List[ScraperStreamingSynapse]):
        default_val_score_responses = [{} for _ in responses]

        try:
            non_fetched_links = {}
            start_time = time.time()

            responses_random_links = [[] for _ in responses]
            all_links = []

            for response, random_links in zip(responses, responses_random_links):
                if not response_uses_twitter_tool(response):
                    continue

                if response.miner_tweets:
                    sample_links = self._sample_cited_and_other_tweets(response)
                    all_links.extend(sample_links)
                    random_links.extend(sample_links)

            unique_links = list(set(all_links))

            if len(unique_links) == 0:
                bt.logging.info("No unique links found to process.")
                return default_val_score_responses

            bt.logging.info(f"Fetching {len(unique_links)} unique Twitter links.")

            tweets_list, non_fetched_links = await scrape_tweets_with_retries(
                unique_links, group_size=200, max_attempts=4
            )

            for response, random_links in zip(responses, responses_random_links):
                ids = [
                    self.tw_client.utils.extract_tweet_id(link) for link in random_links
                ]

                for tweet in tweets_list:
                    if tweet.id in ids:
                        response.validator_tweets.append(tweet)

            end_time = time.time()
            bt.logging.info(
                f"Fetched Twitter links method took {end_time - start_time} seconds. "
                f"All links count: {len(all_links)}, Unique links count: {len(unique_links)}, "
                f"APIFY fetched tweets links count: {len(tweets_list)}"
            )

            bt.logging.info(
                f"Twitter Links not fetched Amount: {len(non_fetched_links)}; List: {non_fetched_links}"
            )
            if len(non_fetched_links):
                bt.logging.info(
                    f"Unique Twitter Links Amount: {len(unique_links)}; List: {unique_links};"
                )

            batch_results = await self.process_response_items_in_batches(
                responses=responses,
                batch_size=20,
                process_function=self.llm_process_validator_tweets,
            )

            val_score_responses_list = [r[0] for r in batch_results]
            durations = [r[1] for r in batch_results if r[1] > 0]
            if durations:
                bt.logging.info(
                    f"[Reward {self.name}] LLM validator tweets: "
                    f"{len(durations)} batches | "
                    f"total={sum(durations) / 60:.2f}min "
                    f"mean={sum(durations) / len(durations):.2f}s "
                    f"min={min(durations):.2f}s max={max(durations):.2f}s"
                )

            return val_score_responses_list
        except Exception as e:
            bt.logging.error(f"Error in process_tweets: {str(e)}")
            return default_val_score_responses

    def check_tweet_content(self, response: ScraperStreamingSynapse):
        try:
            if not response_uses_twitter_tool(response):
                return 0

            tweet_score = 0

            completion = self.get_successful_twitter_completion(response=response)
            if not completion:
                return 0

            tweets_data = response.miner_tweets
            tweets_amount = len(tweets_data)

            if tweets_amount < 2 or not response.validator_tweets:
                # Ensure there are at least two twitter links provided by miners and check for the presence of miner and validator tweets
                return 0

            # Initialize a list to hold scores for each validator tweet
            tweet_scores = []
            # Iterate over all validator tweets instead of selecting a random one
            for val_tweet in response.validator_tweets:
                # Extract content, ID, and creation time of the validator tweet
                val_tweet_content = val_tweet.text
                val_tweet_id = val_tweet.id
                val_tweet_created_at = val_tweet.created_at

                # Find the corresponding miner tweet by ID
                tweet = next(
                    (tweet for tweet in tweets_data if tweet["id"] == val_tweet_id),
                    None,
                )

                # Initialize the score for this iteration
                tweet_score = 0

                if tweet:
                    if not is_valid_tweet(tweet):
                        tweet_scores.append(0)
                        continue

                    tweet_text = tweet["text"]

                    if not tweet_text or re.search(
                        pattern_to_check, tweet_text, flags=re.IGNORECASE
                    ):
                        tweet_scores.append(0)
                        continue

                    # Prepare texts for comparison by normalizing them
                    miner_text_compared = format_text_for_match(tweet_text)
                    validator_text_compared = format_text_for_match(val_tweet_content)

                    if miner_text_compared == validator_text_compared:
                        tweet_score = 1
                    else:
                        tweet_score = 0

                    converted_val_tweet_created_at = (
                        datetime.strptime(
                            val_tweet_created_at, "%a %b %d %H:%M:%S %z %Y"
                        )
                        .astimezone(pytz.UTC)
                        .strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3]
                        + "Z"
                    )

                    if not tweet.get("created_at") == converted_val_tweet_created_at:
                        tweet_score = 0

                    tweet_created_at_aware = datetime.strptime(
                        converted_val_tweet_created_at, "%Y-%m-%dT%H:%M:%S.%fZ"
                    ).replace(tzinfo=pytz.UTC, second=0, microsecond=0)

                    start_date = response.start_date
                    end_date = response.end_date

                    if start_date:
                        start_date = datetime.strptime(
                            start_date, "%Y-%m-%dT%H:%M:%SZ"
                        ).replace(tzinfo=pytz.utc)
                        if tweet_created_at_aware < start_date:
                            tweet_score = 0
                    if end_date:
                        end_date = datetime.strptime(
                            end_date, "%Y-%m-%dT%H:%M:%SZ"
                        ).replace(tzinfo=pytz.utc)
                        if tweet_created_at_aware > end_date:
                            tweet_score = 0

                tweet_scores.append(tweet_score)

            if tweet_scores:
                # Calculate the average score
                average_score = sum(tweet_scores) / len(tweet_scores)
            else:
                # If there are no scores, set average score to 0
                average_score = 0

            return average_score
        except Exception as e:
            bt.logging.error(f"check_tweet_content: {str(e)}")
            return 0

    async def get_rewards(
        self, responses: List[ScraperStreamingSynapse], uids
    ) -> List[BaseRewardEvent]:
        try:
            bt.logging.debug(
                f"TwitterContentRelevanceModel | Calculating {len(responses)} rewards (typically < 1 sec/reward)."
            )

            val_score_responses_list = await self.process_tweets(responses=responses)
            bt.logging.info(f"VAL_SCORE_RESPONSES: {val_score_responses_list}")

            scores = [self.check_tweet_content(response) for response in responses]

            reward_events = []
            scoring_prompt = TweetRelevancePrompt()

            grouped_val_score_responses = []
            missing_validator_tweets = []

            for apify_score, response, val_score_responses, uid_tensor in zip(
                scores,
                responses,
                val_score_responses_list,
                uids,
            ):
                reward_event = BaseRewardEvent()
                reward_event.reward = 0
                is_applicable = response_uses_twitter_tool(response)

                score_result = None
                response_scores = {}
                total_score = 0

                unique_tweet_texts = {}
                for val_tweet in response.validator_tweets:
                    text = format_text_for_match(val_tweet.text)
                    if text not in unique_tweet_texts:
                        unique_tweet_texts[text] = val_tweet

                unique_validator_tweets = list(unique_tweet_texts.values())

                response.validator_tweets = unique_validator_tweets

                if len(response.validator_tweets):
                    missing_validator_tweets.append(0)
                    for val_tweet in response.validator_tweets:
                        val_tweet_id = val_tweet.id
                        if val_score_responses:
                            score_result = val_score_responses.get(
                                str(val_tweet_id), None
                            )
                            if score_result is not None:
                                total_score += (
                                    scoring_prompt.extract_score(score_result) / 3.0
                                )
                                response_scores[val_tweet_id] = (
                                    scoring_prompt.contextual_relevance(score_result)
                                )

                    if total_score > 0:
                        average_score = (
                            total_score / APIFY_LINK_SCRAPE_AMOUNT * apify_score
                        )

                        reward_event.reward = self.clamp_relevance_score(average_score)
                else:
                    missing_validator_tweets.append(1 if is_applicable else 0)
                    reward_event.reward = 0
                reward_events.append(reward_event)
                grouped_val_score_responses.append(response_scores)

            log_reward_aggregates(
                name=self.name,
                uids=uids,
                scores=[e.reward for e in reward_events],
                extras={"missing_val_tweets": missing_validator_tweets},
            )
            return reward_events, grouped_val_score_responses
        except Exception as e:
            error_message = f"Link Content Relevance get_rewards: {str(e)}"
            tb_str = traceback.format_exception(type(e), e, e.__traceback__)
            bt.logging.error("\n".join(tb_str) + error_message)
            reward_events = []
            for response in responses:
                reward_event = BaseRewardEvent()
                reward_event.reward = 0
                reward_events.append(reward_event)
            return reward_events, {}
