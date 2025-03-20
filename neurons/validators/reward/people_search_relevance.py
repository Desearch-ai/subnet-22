import re
import traceback
import time
import random
from typing import List, Dict, Tuple
import json
import asyncio
import bittensor as bt
from .config import RewardModelType
from .reward import BaseRewardModel, BaseRewardEvent
from datura.protocol import PeopleSearchSynapse, PeopleSearchResult
from neurons.validators.apify.linkedin_scraper_actor import LinkedinScraperActor
from datura.utils import is_valid_linkedin_profile

APIFY_LINK_SCRAPE_AMOUNT = 2

EXACT_MATCH_FIELDS = [
    "link",
    "first_name",
    "last_name",
    "full_name",
    "title",
    "summary",
    "avatar",
]

LIST_MATCH_FIELDS = ["experiences", "educations", "languages"]


class PeopleSearchRelevanceModel(BaseRewardModel):
    @property
    def name(self) -> str:
        return RewardModelType.twitter_content_relevance.value

    def __init__(self, device: str, scoring_type: None):
        super().__init__()
        self.device = device
        self.scoring_type = scoring_type
        self.linkedin_scraper_actor = LinkedinScraperActor()

    async def scrape_profiles_with_retries(self, urls, group_size=150, max_attempts=2):
        fetched_profiles = []
        non_fetched_profiles = urls.copy()
        attempt = 1

        while attempt <= max_attempts and non_fetched_profiles:
            bt.logging.info(
                f"Attempt {attempt}/{max_attempts} for processing {len(non_fetched_profiles)} profiles."
            )

            url_groups = [
                non_fetched_profiles[i : i + group_size]
                for i in range(0, len(non_fetched_profiles), group_size)
            ]

            tasks = [
                asyncio.create_task(
                    self.linkedin_scraper_actor.get_profiles(urls=group)
                )
                for group in url_groups
            ]

            # Wait for tasks to complete
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Combine results and handle exceptions
            for result in results:
                if isinstance(result, Exception):
                    bt.logging.error(f"Error attempt {attempt}: {str(result)}")
                    continue
                fetched_profiles.extend(result)

            # Update non-fetched links
            fetched_urls = {link.get("link") for link in fetched_profiles}
            non_fetched_profiles = [
                url for url in non_fetched_profiles if url not in fetched_urls
            ]

            attempt += 1

        return fetched_profiles, non_fetched_profiles

    async def process_profiles(self, responses: List[PeopleSearchSynapse]):
        default_val_score_responses = [{} for _ in responses]

        start_time = time.time()

        all_links = []
        responses_random_profiles = [[] for _ in responses]

        for response, random_profiles in zip(responses, responses_random_profiles):
            urls = [result["link"] for result in response.results if "link" in result]

            if urls:
                sample_links = random.sample(
                    urls,
                    min(APIFY_LINK_SCRAPE_AMOUNT, len(urls)),
                )

                random_profiles.extend(sample_links)
                all_links.extend(sample_links)

        unique_links = list(set(all_links))

        if len(unique_links) == 0:
            bt.logging.info("No unique profiles found to process.")
            return default_val_score_responses

        bt.logging.info(f"Fetching {len(unique_links)} unique profiles.")

        fetched_profiles, non_fetched_profiles = (
            await self.scrape_profiles_with_retries(unique_links)
        )

        for response, random_profiles in zip(responses, responses_random_profiles):
            for profile in fetched_profiles:
                url = profile.get("link")

                if url in random_profiles:
                    response.validator_results.append(PeopleSearchResult(**profile))

        end_time = time.time()
        bt.logging.info(
            f"Fetched profiles method took {end_time - start_time} seconds. "
            f"All profiles count: {len(all_links)}, Unique profiles count: {len(unique_links)}, "
            f"APIFY fetched profiles count: {len(fetched_profiles)}"
        )

        bt.logging.info(
            f"Profiles not fetched amount: {len(non_fetched_profiles)}; List: {non_fetched_profiles}"
        )
        if len(non_fetched_profiles):
            bt.logging.info(
                f"Unique Profiles Amount: {len(unique_links)}; List: {unique_links};"
            )

        return default_val_score_responses

    def compare_lists(self, list1, list2) -> bool:
        if not isinstance(list1, list) or not isinstance(list2, list):
            return False

        set1 = {frozenset(item.items()) for item in list1}
        set2 = {frozenset(item.items()) for item in list2}

        return set1 == set2

    def check_relevance(self, response: PeopleSearchSynapse) -> float:
        return 1.0

    def check_response(self, response: PeopleSearchSynapse) -> float:
        try:
            # 1) Gather miner & validator tweets
            miner_results = response.results
            validator_results = response.validator_results

            # 2) Build map of miner tweets by ID
            miner_map = {}

            for profile in miner_results:
                if "link" in profile:
                    if miner_map.get(profile["link"]):
                        return 0.0
                    else:
                        miner_map[profile["link"]] = profile

            scores = []

            # 3) Iterate over validator tweets
            for val_profile in validator_results:
                # Match miner tweet by ID
                if not val_profile.link or val_profile.link not in miner_map:
                    scores.append(0)
                    continue

                miner_profile = miner_map[val_profile.link]

                if not is_valid_linkedin_profile(miner_profile):
                    scores.append(0)
                    continue

                val_profile_data = val_profile.model_dump()

                loop_terminated = False
                for f in EXACT_MATCH_FIELDS:
                    if miner_profile.get(f) != val_profile_data.get(f):
                        scores.append(0)
                        bt.logging.debug(
                            f"Field mismatch: {f} => {miner_profile.get(f)} vs {val_profile_data.get(f)}"
                        )
                        loop_terminated = True
                        break
                if loop_terminated:
                    continue

                for f in LIST_MATCH_FIELDS:
                    if not self.compare_lists(
                        miner_profile.get(f), val_profile_data.get(f)
                    ):
                        scores.append(0)
                        bt.logging.debug(
                            f"Field mismatch: {f} => {miner_profile.get(f)} vs {val_profile_data.get(f)}"
                        )
                        loop_terminated = True
                        break
                if loop_terminated:
                    continue

                # All checks passed => score = 1
                scores.append(self.check_relevance(response))

            # Return average of all validated profiles
            return sum(scores) / len(scores) if scores else 0.0

        except Exception as e:
            bt.logging.error(f"check_people_response error: {str(e)}")
            return 0.0

    async def get_rewards(
        self, responses: List[PeopleSearchSynapse], uids: List[int]
    ) -> Tuple[List[BaseRewardEvent], Dict[int, float]]:
        try:
            # Step 1: fetch and fill validator_links
            _ = await self.process_profiles(responses=responses)

            reward_events = []
            zero_scores = {}
            non_zero_scores = {}
            grouped_val_score_responses = {}

            # Step 2: for each response, compute a final score
            for response, uid_tensor in zip(responses, uids):
                # If uid_tensor is a PyTorch or NumPy scalar, .item() extracts the integer
                uid = uid_tensor.item() if hasattr(uid_tensor, "item") else uid_tensor

                final_score = self.check_response(response)

                bt.logging.info(
                    f"UID {uid}: check_response_random_link => {final_score}"
                )

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
                f"========== Profile Relevance Zero Scores ({len(zero_scores)} cases) =========="
            )
            bt.logging.info(json.dumps(zero_scores))
            bt.logging.info(
                f"======== Profile Relevance Non-Zero Scores ({len(non_zero_scores)} cases) ========"
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
