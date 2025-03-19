import re
import traceback
import time
import random
from typing import List, Dict, Tuple
import json
import asyncio
import html
import bittensor as bt
from .config import RewardModelType
from .reward import BaseRewardModel, BaseRewardEvent
from datura.protocol import WebSearchSynapse, WebSearchValidatorResult
from datura.services.twitter_utils import TwitterUtils
from datura.utils import is_valid_web_search_result
from neurons.validators.apify.cheerio_scraper_actor import CheerioScraperActor
from neurons.validators.apify.reddit_scraper_actor import RedditScraperActor

APIFY_LINK_SCRAPE_AMOUNT = 1


class WebBasicSearchContentRelevanceModel(BaseRewardModel):
    @property
    def name(self) -> str:
        return RewardModelType.twitter_content_relevance.value

    def __init__(self, device: str, scoring_type: None):
        super().__init__()
        self.device = device
        self.scoring_type = scoring_type

    def normalize_html_content(self, content: str) -> str:
        if content is None:
            return ""

        normalized_content = re.sub(
            r"\s+", " ", content.replace("\n", " ").replace("\r", " ").strip()
        )
        return html.unescape(normalized_content).lower()

    async def scrape_with_retries(
        self, urls, scraper_actor_class, group_size, max_attempts
    ):
        fetched_links_with_metadata = []
        non_fetched_links = urls.copy()
        attempt = 1

        while attempt <= max_attempts and non_fetched_links:
            bt.logging.info(
                f"Attempt {attempt}/{max_attempts} for {scraper_actor_class.__name__}, processing {len(non_fetched_links)} links."
            )

            url_groups = [
                non_fetched_links[i : i + group_size]
                for i in range(0, len(non_fetched_links), group_size)
            ]

            tasks = [
                asyncio.create_task(scraper_actor_class().scrape_metadata(urls=group))
                for group in url_groups
            ]

            # Wait for tasks to complete
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Combine results and handle exceptions
            for result in results:
                if isinstance(result, Exception):
                    bt.logging.error(
                        f"Error in {scraper_actor_class.__name__} scraper attempt {attempt}: {str(result)}"
                    )
                    continue
                fetched_links_with_metadata.extend(result)

            # Update non-fetched links
            fetched_urls = {link.get("link") for link in fetched_links_with_metadata}
            non_fetched_links = [
                url for url in non_fetched_links if url not in fetched_urls
            ]

            attempt += 1

        return fetched_links_with_metadata, non_fetched_links

    async def scrape_links_with_retries(self, urls):
        # Separate Reddit URLs from other URLs
        reddit_urls = []
        other_urls = []

        for url in urls:
            if "reddit.com" in url and "comments" in url:
                reddit_urls.append(url)
            else:
                other_urls.append(url)

        # Scrape Reddit URLs with retries
        reddit_fetched_links_with_metadata = []
        reddit_non_fetched_links = []

        if reddit_urls:
            reddit_fetched_links_with_metadata, reddit_non_fetched_links = (
                await self.scrape_with_retries(
                    urls=reddit_urls,
                    scraper_actor_class=RedditScraperActor,
                    group_size=200,
                    max_attempts=2,
                )
            )

        # Scrape other URLs with retries
        other_fetched_links_with_metadata = []
        other_non_fetched_links = []

        if other_urls:
            other_fetched_links_with_metadata, other_non_fetched_links = (
                await self.scrape_with_retries(
                    urls=other_urls,
                    scraper_actor_class=CheerioScraperActor,
                    group_size=100,
                    max_attempts=2,
                )
            )

        # Combine non-fetched links
        non_fetched_links = reddit_non_fetched_links + other_non_fetched_links

        # Combine all fetched links
        fetched_links_with_metadata = (
            reddit_fetched_links_with_metadata + other_fetched_links_with_metadata
        )

        # Filter out any entries without a URL
        fetched_links_with_metadata = [
            link for link in fetched_links_with_metadata if link.get("link")
        ]

        return fetched_links_with_metadata, non_fetched_links

    async def process_links(self, responses: List[WebSearchSynapse]):
        default_val_score_responses = [{} for _ in responses]

        start_time = time.time()

        all_links = []
        responses_random_links = [[] for _ in responses]

        for response, random_links in zip(responses, responses_random_links):
            urls = [result["link"] for result in response.results if "link" in result]

            if urls:
                sample_links = random.sample(
                    urls,
                    min(APIFY_LINK_SCRAPE_AMOUNT, len(urls)),
                )

                random_links.extend(sample_links)
                all_links.extend(sample_links)

        unique_links = list(set(all_links))

        if len(unique_links) == 0:
            bt.logging.info("No unique links found to process.")
            return default_val_score_responses

        bt.logging.info(f"Fetching {len(unique_links)} unique web links.")

        links_with_metadata, non_fetched_links = await self.scrape_links_with_retries(
            unique_links
        )

        for response, random_links in zip(responses, responses_random_links):
            for link_with_metadata in links_with_metadata:
                url = link_with_metadata.get("link")

                if url in random_links:
                    response.validator_links.append(
                        WebSearchValidatorResult(**link_with_metadata)
                    )

        end_time = time.time()
        bt.logging.info(
            f"Fetched Web links method took {end_time - start_time} seconds. "
            f"All links count: {len(all_links)}, Unique links count: {len(unique_links)}, "
            f"APIFY fetched web links count: {len(links_with_metadata)}"
        )

        bt.logging.info(
            f"Web links not fetched amount: {len(non_fetched_links)}; List: {non_fetched_links}"
        )
        if len(non_fetched_links):
            bt.logging.info(
                f"Unique Web Links Amount: {len(unique_links)}; List: {unique_links};"
            )

        return default_val_score_responses

    def check_title(self, miner_title, validator_title):
        miner_title = miner_title.rstrip(" .")

        if miner_title in validator_title or validator_title in miner_title:
            return True

        return False

    def check_response_random_link(self, response: WebSearchSynapse) -> float:
        try:
            miner_results = response.results
            validator_links = response.validator_links

            miner_map = {}

            for miner_item in miner_results:
                if "link" in miner_item:
                    if miner_map.get(miner_item["link"]):
                        return 0.0
                    else:
                        miner_map[miner_item["link"]] = miner_item

            scores = []

            for validator_item in validator_links:
                if not validator_item.link or validator_item.link not in miner_map:
                    scores.append(0)
                    continue

                miner_item = miner_map[validator_item.link]

                if not is_valid_web_search_result(miner_item):
                    scores.append(0)
                    continue

                if not self.check_title(miner_item.get("title"), validator_item.title):
                    scores.append(0)
                    continue

                if miner_item.get("link") != validator_item.link:
                    scores.append(0)
                    continue

                if not all(
                    text.strip()
                    in self.normalize_html_content(validator_item.html_content)
                    or text.strip()
                    in self.normalize_html_content(validator_item.html_text)
                    for text in re.split(r"[.·]", miner_item.get("snippet").lower())
                ):
                    scores.append(0)
                    continue

                query_words = response.query.strip().lower().split(" ")

                texts = [validator_item.title.lower(), validator_item.snippet.lower()]

                if response.query and not any(
                    word in text for word in query_words for text in texts
                ):
                    scores.append(0)
                    continue

                scores.append(1)

            return sum(scores) / len(scores) if scores else 0.0
        except Exception as e:
            bt.logging.error(f"check_response_random_link error: {str(e)}")
            return 0.0

    async def get_rewards(
        self, responses: List[WebSearchSynapse], uids: List[int]
    ) -> Tuple[List[BaseRewardEvent], Dict[int, float]]:
        try:
            # Step 1: fetch and fill validator_links
            _ = await self.process_links(responses=responses)

            reward_events = []
            zero_scores = {}
            non_zero_scores = {}
            grouped_val_score_responses = {}

            # Step 2: for each response, compute a final score
            for response, uid_tensor in zip(responses, uids):
                # If uid_tensor is a PyTorch or NumPy scalar, .item() extracts the integer
                uid = uid_tensor.item() if hasattr(uid_tensor, "item") else uid_tensor

                final_score = self.check_response_random_link(response)

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
                f"========== Web Link Content Zero Scores ({len(zero_scores)} cases) =========="
            )
            bt.logging.info(json.dumps(zero_scores))
            bt.logging.info(
                f"======== Web Link Content Non-Zero Scores ({len(non_zero_scores)} cases) ========"
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
