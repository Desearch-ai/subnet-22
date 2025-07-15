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
# DEALINGS IN THE SOFTWARE.

import traceback
import time
import random
from typing import List, Optional, Dict, Any, Tuple
import json
from datetime import datetime
import pytz
import bittensor as bt

from neurons.validators.base_validator import AbstractNeuron
from .config import RewardModelType
from .reward import BaseRewardModel, BaseRewardEvent
from datura.protocol import (
    TwitterSearchSynapse,
    TwitterIDSearchSynapse,
    TwitterURLsSearchSynapse,
)
from datura.services.twitter_utils import TwitterUtils
from datura.utils import (
    clean_text,
    format_text_for_match,
    is_valid_tweet,
    scrape_tweets_with_retries,
)

APIFY_LINK_SCRAPE_AMOUNT = 1

# Only a percentage-based threshold:
INT_DIFF_PERCENT = 0.60  # 60% difference allowed


TWEET_EXACT_MATCH_FIELDS = {
    "id",
    "url",
    "created_at",
    "is_quote_tweet",
    "is_retweet",
    "conversation_id",
    # "in_reply_to_screen_name",
    "in_reply_to_status_id",
    # "in_reply_to_user_id",
    "quoted_status_id",
    "lang",
}

USER_EXACT_FIELDS = {
    "id",
    "url",
    "name",
    "username",
    "created_at",
    "description",
    "profile_image_url",
    "profile_banner_url",
    "verified",
    "can_dm",
    "can_media_tag",
    "location",
    "pinned_tweet_ids",
    "is_blue_verified",
}

TWEET_NUMERIC_FIELDS = {
    "reply_count",
    "retweet_count",
    "like_count",
    "quote_count",
    "bookmark_count",
}

USER_NUMERIC_FIELDS = {
    # "favourites_count",
    "followers_count",
    "media_count",
    "statuses_count",
}

TWEET_NESTED_FIELDS = {"quote", "entities", "extended_entities"}

USER_NESTED_FIELDS = {"entities"}


class TwitterBasicSearchContentRelevanceModel(BaseRewardModel):
    @property
    def name(self) -> str:
        return RewardModelType.twitter_basic_search_content_relevance.value

    def __init__(self, device: str, scoring_type: None, neuron: AbstractNeuron):
        super().__init__(neuron)
        self.device = device
        self.scoring_type = scoring_type
        self.twitter_utils = TwitterUtils()

    def clean_text(self, text):
        return clean_text(text)

    async def process_tweets(self, responses: List[TwitterSearchSynapse]):
        default_val_score_responses = [{} for _ in responses]

        try:
            start_time = time.time()
            responses_random_links = [[] for _ in responses]
            all_links = []

            # 1) Collect & sample URLs from each synapse.results
            for response, random_links in zip(responses, responses_random_links):
                tweet_urls = [
                    tweet["url"] for tweet in response.results if "url" in tweet
                ]

                if tweet_urls:
                    sample_links = random.sample(
                        tweet_urls,
                        min(APIFY_LINK_SCRAPE_AMOUNT, len(tweet_urls)),
                    )
                    all_links.extend(sample_links)
                    random_links.extend(sample_links)

            unique_links = list(set(all_links))
            if len(unique_links) == 0:
                bt.logging.info("No unique links found to process (no tweet URLs).")
                return default_val_score_responses

            bt.logging.info(f"Fetching {len(unique_links)} unique Twitter links.")
            tweets_list, non_fetched_links = await scrape_tweets_with_retries(
                unique_links, group_size=200, max_attempts=4
            )

            # 2) For each response, match tweets by ID and append to validator_tweets
            for response, random_links in zip(responses, responses_random_links):
                ids = [
                    self.twitter_utils.extract_tweet_id(link) for link in random_links
                ]
                for fetched_tweet in tweets_list:
                    if fetched_tweet.id in ids:
                        # Append the newly fetched tweet to validator_tweets
                        response.validator_tweets.append(fetched_tweet)

            end_time = time.time()
            bt.logging.info(
                f"Fetched Twitter links took {end_time - start_time:.2f}s. "
                f"All links: {len(all_links)}, Unique: {len(unique_links)}, "
                f"Fetched: {len(tweets_list)}"
            )
            bt.logging.info(
                f"Non-fetched count: {len(non_fetched_links)}, List: {non_fetched_links}"
            )

            return default_val_score_responses

        except Exception as e:
            bt.logging.error(f"Error in process_tweets: {str(e)}")
            return default_val_score_responses

    def compare_numeric(
        self, field: str, val1: Optional[int], val2: Optional[int]
    ) -> bool:
        """
        Returns True if the absolute difference between numeric values
        is within the specified percentage threshold of the validator_value.
        """

        if val1 is None and val2 is None:
            return True

        if val1 is None or val2 is None:
            return False

        allowed_diff = max(int(val2 * INT_DIFF_PERCENT), 10)

        diff = abs(val1 - val2)
        is_allowed = diff <= allowed_diff

        if not is_allowed:
            bt.logging.debug(
                f"{field} value mismatch: {val1} vs {val2}, allowed: {allowed_diff}, diff: {diff}"
            )

        return is_allowed

    def compare_nested_fields(
        self,
        val1: Optional[Dict[str, Any]],
        val2: Optional[Dict[str, Any]],
        path: Optional[str] = "",
    ) -> Tuple[str, Any, Any]:
        """
        Returns True if all the nested fields within the values are equal.
        """

        if val1 is None and val2 is None:
            return "", None, None

        if val1 is None or val2 is None:
            return path, val1, val2

        if isinstance(val1, dict) and isinstance(val2, dict):
            for key in set(val1) | set(val2):
                _path, _val1, _val2 = self.compare_nested_fields(
                    val1.get(key), val2.get(key), f"{path}.{key}"
                )
                if _path:
                    return _path, _val1, _val2

            return "", None, None

        if isinstance(val1, list) and isinstance(val2, list):
            if len(val1) != len(val2):
                return path, val1, val2

            for i, (x, y) in enumerate(zip(val1, val2)):
                _path, _val1, _val2 = self.compare_nested_fields(x, y, f"{path}[{i}]")
                if _path:
                    return _path, _val1, _val2

            return "", None, None

        if isinstance(val1, tuple) and isinstance(val2, tuple):
            if len(val1) != len(val2):
                return path, val1, val2

            for i, (x, y) in enumerate(zip(val1, val2)):
                _path, _val1, _val2 = self.compare_nested_fields(x, y, f"{path}[{i}]")
                if _path:
                    return _path, _val1, _val2

            return "", None, None

        return ("", None, None) if val1 == val2 else (path, val1, val2)

    def compare_media(self, media1: List[dict], media2: List[dict]) -> bool:
        if len(media1) != len(media2):
            return False

        return all(
            m1.get("type") == m2.get("type")
            and m1.get("media_url") == m2.get("media_url")
            for m1, m2 in zip(media1, media2)
        )

    def compare_content(self, text1: str, text2: str) -> bool:
        return format_text_for_match(text1) == format_text_for_match(text2)

    def check_tweet_content(
        self,
        response: (
            TwitterSearchSynapse | TwitterIDSearchSynapse | TwitterURLsSearchSynapse
        ),
    ) -> float:
        try:
            # 1) Gather miner & validator tweets
            miner_data_list = response.results
            validator_tweets = response.validator_tweets

            # 2) Build map of miner tweets by ID
            miner_map = {}

            for tweet_dict in miner_data_list:
                if "id" in tweet_dict:
                    if miner_map.get(tweet_dict["id"]):
                        return 0.0
                    else:
                        miner_map[tweet_dict["id"]] = tweet_dict

            tweet_scores = []

            # 3) Iterate over validator tweets
            for val_tweet in validator_tweets:
                # Match miner tweet by ID
                if not val_tweet.id or val_tweet.id not in miner_map:
                    tweet_scores.append(0)
                    continue

                miner_tweet = miner_map[val_tweet.id]

                if not is_valid_tweet(miner_tweet):
                    tweet_scores.append(0)
                    continue

                # b) If it's TwitterIDSearchSynapse => confirm val_tweet.id == response.id
                if isinstance(response, TwitterIDSearchSynapse):
                    if val_tweet.id != response.id:
                        tweet_scores.append(0)
                        continue

                # c) If it's TwitterURLsSearchSynapse => confirm val_tweet.url is in response.urls
                if isinstance(response, TwitterURLsSearchSynapse):
                    if not val_tweet.url or (val_tweet.url not in response.urls):
                        tweet_scores.append(0)
                        continue

                tweet_score = []
                # d) If it's TwitterSearchSynapse => check min_likes/min_retweets/min_replies
                if isinstance(response, TwitterSearchSynapse):
                    synapse = response.copy(deep=True)
                    query = response.query.strip().lower()

                    if "from:" in query:
                        try:
                            synapse.user = query.split("from:")[1].split(" ")[0].strip()
                        except:
                            pass

                    if "min_faves:" in query:
                        try:
                            synapse.min_likes = int(
                                query.split("min_faves:")[1].split(" ")[0].strip()
                            )
                        except:
                            pass

                    if "min_retweets:" in query:
                        try:
                            synapse.min_retweets = int(
                                query.split("min_retweets:")[1].split(" ")[0].strip()
                            )
                        except:
                            pass

                    if "min_replies:" in query:
                        try:
                            synapse.min_replies = int(
                                query.split("min_replies:")[1].split(" ")[0].strip()
                            )
                        except:
                            pass

                    if "filter:verified" in query:
                        synapse.verified = True

                    if "filter:blue_verified" in query:
                        synapse.blue_verified = True

                    if "filter:quote" in query:
                        synapse.is_quote = True

                    if "filter:images" in query:
                        synapse.is_image = True

                    if "filter:videos" in query:
                        synapse.is_video = True

                    if "since:" in query:
                        try:
                            synapse.start_date = (
                                query.split("since:")[1].split(" ")[0].strip()
                            )
                        except:
                            pass

                    if "until:" in query:
                        try:
                            synapse.end_date = (
                                query.split("until:")[1].split(" ")[0].strip()
                            )
                        except:
                            pass

                    if "lang:" in query:
                        try:
                            synapse.lang = query.split("lang:")[1].split(" ")[0].strip()
                        except:
                            pass

                    query_words = synapse.query.strip().lower().split(" ")

                    texts = [
                        val_tweet.text.lower(),
                        val_tweet.user.username.lower(),
                        val_tweet.user.name.lower(),
                    ]

                    # Check any of query words to be in tweet text
                    if synapse.query and not any(
                        word in text for word in query_words for text in texts
                    ):
                        tweet_score.append(0)
                    else:
                        tweet_score.append(1)

                    if synapse.min_likes is not None:
                        if (
                            val_tweet.like_count is None
                            or val_tweet.like_count < synapse.min_likes
                        ):
                            tweet_score.append(0)
                        else:
                            tweet_score.append(1)

                    if synapse.min_retweets is not None:
                        if (
                            val_tweet.retweet_count is None
                            or val_tweet.retweet_count < synapse.min_retweets
                        ):
                            tweet_score.append(0)
                        else:
                            tweet_score.append(1)

                    if synapse.min_replies is not None:
                        if (
                            val_tweet.reply_count is None
                            or val_tweet.reply_count < synapse.min_replies
                        ):
                            tweet_score.append(0)
                        else:
                            tweet_score.append(1)

                    if synapse.user is not None:
                        if synapse.user != val_tweet.user.username:
                            tweet_score.append(0)
                        else:
                            tweet_score.append(1)

                    if synapse.verified is not None:
                        if synapse.verified != val_tweet.user.verified:
                            tweet_score.append(0)
                        else:
                            tweet_score.append(1)

                    if synapse.is_quote is not None:
                        if synapse.is_quote != val_tweet.is_quote_tweet:
                            tweet_score.append(0)
                        else:
                            tweet_score.append(1)

                    if synapse.is_image is not None:
                        has_image_media = any(
                            m.type == "photo" for m in val_tweet.media
                        )

                        if synapse.is_image != has_image_media:
                            tweet_score.append(0)
                        else:
                            tweet_score.append(1)

                    if synapse.is_video is not None:
                        has_video_media = any(
                            m.type == "video" for m in val_tweet.media
                        )

                        if synapse.is_video != has_video_media:
                            tweet_score.append(0)
                        else:
                            tweet_score.append(1)

                    tweet_date = datetime.strptime(
                        val_tweet.created_at, "%a %b %d %H:%M:%S %z %Y"
                    ).replace(tzinfo=pytz.UTC)

                    if synapse.start_date is not None:
                        try:
                            start_date = datetime.strptime(
                                synapse.start_date, "%Y-%m-%d_%H:%M:%S_%Z"
                            ).replace(tzinfo=pytz.UTC)
                        except ValueError:
                            start_date = datetime.strptime(
                                synapse.start_date, "%Y-%m-%d"
                            ).replace(tzinfo=pytz.UTC)

                        if tweet_date < start_date:
                            tweet_score.append(0)
                        else:
                            tweet_score.append(1)

                    if synapse.end_date is not None:
                        try:
                            end_date = datetime.strptime(
                                synapse.end_date, "%Y-%m-%d_%H:%M:%S_%Z"
                            ).replace(tzinfo=pytz.UTC)
                        except ValueError:
                            end_date = datetime.strptime(
                                synapse.end_date, "%Y-%m-%d"
                            ).replace(tzinfo=pytz.UTC)

                        if tweet_date > end_date:
                            tweet_score.append(0)
                        else:
                            tweet_score.append(1)

                    if synapse.lang is not None:
                        if synapse.lang != val_tweet.lang:
                            tweet_score.append(0)
                        else:
                            tweet_score.append(1)

                    if synapse.blue_verified is not None:
                        if synapse.blue_verified != val_tweet.user.is_blue_verified:
                            tweet_score.append(0)
                        else:
                            tweet_score.append(1)

                val_tweet_dict = val_tweet.model_dump()

                # # Compare tweet basic fields
                for f in TWEET_EXACT_MATCH_FIELDS:
                    if miner_tweet.get(f) != val_tweet_dict.get(f):
                        tweet_score.append(0)
                        bt.logging.debug(
                            f"Field mismatch: {f} => {miner_tweet.get(f)} vs {val_tweet_dict.get(f)}"
                        )
                    else:
                        tweet_score.append(1)

                if not self.compare_content(
                    miner_tweet.get("text"), val_tweet_dict.get("text")
                ):
                    tweet_score.append(0)
                else:
                    tweet_score.append(1)

                # Compare numeric fields
                for f in TWEET_NUMERIC_FIELDS:
                    if not self.compare_numeric(
                        f, miner_tweet.get(f), val_tweet_dict.get(f)
                    ):
                        tweet_score.append(0)
                        bt.logging.debug(
                            f"Field mismatch: {f} => {miner_tweet.get(f)} vs {val_tweet_dict.get(f)}"
                        )
                    else:
                        tweet_score.append(1)

                # Compare nested fields
                # for f in TWEET_NESTED_FIELDS:
                #     path, val1, val2 = self.compare_nested_fields(
                #         miner_tweet.get(f), val_tweet_dict.get(f)
                #     )
                #     if path:
                #         tweet_scores.append(0)
                #         bt.logging.debug(
                #             f"Field mismatch: {f}{path} => {val1} vs {val2}"
                #         )
                #         loop_terminated = True
                #         break
                # if loop_terminated:
                #     continue

                # Compare media
                if not self.compare_media(
                    miner_tweet.get("media"), val_tweet_dict.get("media")
                ):
                    tweet_score.append(0)
                    bt.logging.debug(
                        f"Tweet media mismatch: {f} => {miner_user.get('media')} vs {val_user.get('media')}"
                    )
                else:
                    tweet_score.append(1)

                miner_user = miner_tweet.get("user")
                val_user = val_tweet_dict.get("user")

                for f in USER_EXACT_FIELDS:
                    if miner_user.get(f) != val_user.get(f):
                        tweet_score.append(0)
                        bt.logging.debug(
                            f"User field mismatch: {f} => {miner_user.get(f)} vs {val_user.get(f)}"
                        )
                    else:
                        tweet_score.append(1)

                for f in USER_NUMERIC_FIELDS:
                    if not self.compare_numeric(f, miner_user.get(f), val_user.get(f)):
                        tweet_score.append(0)
                        bt.logging.debug(
                            f"User field mismatch: {f} => {miner_user.get(f)} vs {val_user.get(f)}"
                        )
                    else:
                        tweet_score.append(1)

                for f in USER_NESTED_FIELDS:
                    path, val1, val2 = self.compare_nested_fields(
                        miner_user.get(f), val_user.get(f)
                    )
                    if path:
                        tweet_score.append(0)
                        bt.logging.debug(
                            f"User field mismatch: {f}{path} => {val1} vs {val2}"
                        )
                    else:
                        tweet_score.append(1)

                # All checks passed => score = 1
                tweet_scores.append(
                    sum(tweet_score) / len(tweet_score) if tweet_score else 0.0
                )

            # Return average of all validated tweets
            return sum(tweet_scores) / len(tweet_scores) if tweet_scores else 0.0

        except Exception as e:
            bt.logging.error(f"check_tweet_content error: {str(e)}")
            return 0.0

    async def get_rewards(
        self, responses: List[TwitterSearchSynapse], uids: List[int]
    ) -> Tuple[List[BaseRewardEvent], Dict[int, float]]:
        try:
            # Step 1: fetch and fill validator_tweets
            _ = await self.process_tweets(responses=responses)

            reward_events = []
            zero_scores = {}
            non_zero_scores = {}
            grouped_val_score_responses = {}

            # Step 2: for each response, compute a final score
            for response, uid_tensor in zip(responses, uids):
                # If uid_tensor is a PyTorch or NumPy scalar, .item() extracts the integer
                uid = uid_tensor.item() if hasattr(uid_tensor, "item") else uid_tensor

                final_score = self.check_tweet_content(response)

                bt.logging.info(f"UID {uid}: check_tweet_content => {final_score}")

                # Step 3: create a reward event
                reward_event = BaseRewardEvent()
                reward_event.reward = final_score
                reward_events.append(reward_event)

                # Keep track of final_score for logging
                if final_score == 0:
                    zero_scores[uid] = final_score
                else:
                    non_zero_scores[uid] = final_score

                # Populate grouped_val_score_responses with final_score
                grouped_val_score_responses[uid] = final_score

            # Step 4: Log zero vs. non-zero
            bt.logging.info(
                f"========== Twitter Link Content Zero Scores ({len(zero_scores)} cases) =========="
            )
            bt.logging.info(json.dumps(zero_scores))
            bt.logging.info(
                f"======== Twitter Link Content Non-Zero Scores ({len(non_zero_scores)} cases) ========"
            )
            bt.logging.info(json.dumps(non_zero_scores))

            return reward_events, grouped_val_score_responses
        except Exception as e:
            tb_str = traceback.format_exception(type(e), e, e.__traceback__)
            bt.logging.error("\n".join(tb_str))

            # On exception, return zeroed events
            reward_events = []
            for _ in responses:
                revent = BaseRewardEvent()
                revent.reward = 0
                reward_events.append(revent)

            return reward_events, {}
