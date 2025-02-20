from typing import List
from .reward import BaseRewardModel, BaseRewardEvent
from .config import RewardModelType
from neurons.validators.reward.reward_llm import RewardLLM
from datura.protocol import ScraperStreamingSynapse
import traceback
import bittensor as bt
from datura.utils import clean_text
from neurons.validators.apify.cheerio_scraper_actor import CheerioScraperActor
from neurons.validators.apify.reddit_scraper_actor import RedditScraperActor
import asyncio
from neurons.validators.utils.prompts import (
    SearchSummaryRelevancePrompt,
)
import random
import json
import time
import math


class WebSearchContentRelevanceModel(BaseRewardModel):
    reward_model_name: str = "VMware/open-llama-7b-open-instruct"

    @property
    def name(self) -> str:
        return RewardModelType.search_content_relevance.value

    def __init__(self, device: str, scoring_type: None, llm_reward: RewardLLM):
        super().__init__()
        self.device = device
        self.reward_llm = llm_reward

        self.scoring_type = scoring_type

    async def llm_process_validator_links(self, response: ScraperStreamingSynapse):
        if not response.validator_links:
            return {}

        scoring_messages = []

        for validator_link in response.validator_links:
            url = validator_link.get("link")
            title = validator_link.get("title", "")
            description = validator_link.get("snippet", "")

            result = self.get_scoring_text(
                prompt=response.prompt,
                content=f"Title: {title}, Description: {description}",
                response=None,
            )
            if result:
                _, scoring_text = result
                scoring_messages.append({url: scoring_text})

        score_responses = await self.reward_llm.llm_processing(scoring_messages)
        return score_responses

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

    async def process_links(self, responses: List[ScraperStreamingSynapse]):
        default_val_score_responses = [{} for _ in responses]

        start_time = time.time()

        all_links = []
        responses_random_links = [[] for _ in responses]

        for response, random_links in zip(responses, responses_random_links):
            # Extract random links from each summary (Search, Reddit, Hacker News)
            completion = self.get_successful_search_summary_completion(response)

            if not completion:
                continue

            _, links_per_summary = response.get_search_links()

            # If scoring single summary 2 link is selected, for 2 or 3 summaries 1 link is selected from each
            random_links_per_summary = 2 if len(links_per_summary) == 1 else 1

            links = []

            for summary_links in links_per_summary.values():
                links.extend(
                    random.sample(
                        summary_links,
                        min(random_links_per_summary, len(summary_links)),
                    )
                )

            random_links.extend(links)
            all_links.extend(links)

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
                    response.validator_links.append(link_with_metadata)

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

        val_score_responses_list = await self.process_response_items_in_batches(
            responses=responses,
            batch_size=20,
            process_function=self.llm_process_validator_links,
        )

        return val_score_responses_list

    def check_response_random_link(self, response: ScraperStreamingSynapse):
        try:
            completion = self.get_successful_search_summary_completion(
                response=response
            )

            if not completion:
                return 0

            search_completion_links = response.search_completion_links
            validator_links = response.validator_links

            if not search_completion_links or not validator_links:
                return 0

            if len(search_completion_links) < 2:
                # at least miners should provide two search links
                return 0

            # Web search results are separate because they include links with different domains from search
            web_search_results = str(response.search_results)

            domain_to_search_result = {
                "arxiv.org": response.arxiv_search_results,
                "wikipedia.org": response.wikipedia_search_results,
                "reddit.com": response.reddit_search_results,
                "ycombinator.com": response.hacker_news_search_results,
                "youtube.com": response.youtube_search_results,
            }

            link_scores = []

            for val_link in validator_links:
                url = val_link.get("link")

                if not url:
                    link_scores.append(0)
                    continue

                domain_parts = url.split("/")[2].split(".")
                domain = ".".join(domain_parts[-2:])  # Extract the main domain

                if domain in domain_to_search_result:
                    if (
                        url in str(domain_to_search_result[domain])
                        or url in web_search_results
                    ):
                        link_scores.append(1)
                    else:
                        link_scores.append(0)
                else:
                    link_scores.append(1 if url in web_search_results else 0)

            if link_scores:
                return sum(link_scores) / len(link_scores)

            return 0
        except Exception as e:
            bt.logging.error(f"check_response_random_link: {str(e)}")
            return 0

    def get_scoring_text(
        self, prompt: str, content: str, response: ScraperStreamingSynapse
    ) -> BaseRewardEvent:
        try:
            if response:
                completion = self.get_successful_search_summary_completion(
                    response=response
                )

                if not completion:
                    return None

            if content is None:
                bt.logging.debug("Search Content is empty.")
                return None

            content = clean_text(content)

            scoring_prompt_text = None
            scoring_prompt = SearchSummaryRelevancePrompt()

            if not scoring_prompt_text:
                scoring_prompt_text = scoring_prompt.text(prompt, content)

            return scoring_prompt, [
                {"role": "system", "content": scoring_prompt.get_system_message()},
                {"role": "user", "content": scoring_prompt_text},
            ]
        except Exception as e:
            bt.logging.error(f"Error in Prompt reward method: {str(e)}")
            return None

    async def get_rewards(
        self, responses: List[ScraperStreamingSynapse], uids
    ) -> List[BaseRewardEvent]:
        try:
            completions: List[str] = self.get_successful_search_completions(responses)
            bt.logging.debug(
                f"WebSearchContentRelevanceModel | Calculating {len(completions)} rewards (typically < 1 sec/reward)."
            )

            val_score_responses_list = await self.process_links(responses=responses)

            scores = [
                self.check_response_random_link(response) for response in responses
            ]

            reward_events = []
            scoring_prompt = SearchSummaryRelevancePrompt()

            grouped_val_score_responses = []

            for apify_score, response, val_score_responses, uid_tensor in zip(
                scores, responses, val_score_responses_list, uids
            ):
                uid = uid_tensor.item()

                #  If only the Twitter tool is used, automatically assign reward=1 and skip normal scoring
                if len(response.tools) == 1 and "Twitter Search" in response.tools:
                    reward_event = BaseRewardEvent()
                    reward_event.reward = 1.0
                    reward_events.append(reward_event)
                    # You might append an empty dict or skip for the grouped responses
                    grouped_val_score_responses.append({})
                    continue

                reward_event = BaseRewardEvent()
                reward_event.reward = 0

                response_scores = {}
                total_score = 0
                num_links = len(response.validator_links)
                _, links_expected = response.get_search_completion()

                max_links_considered = max(num_links, links_expected)

                if num_links > 0:
                    for val_link in response.validator_links:
                        val_url = val_link.get("link")
                        if val_score_responses:
                            score_result = val_score_responses.get(val_url, None)
                            if score_result is not None:
                                score = scoring_prompt.extract_score(score_result)
                                total_score += score / 10.0
                                response_scores[val_url] = score

                    if total_score > 0:
                        average_score = total_score / max_links_considered
                        reward_event.reward = self.calculate_adjusted_score(
                            links_count=len(response.search_completion_links),
                            score=average_score,
                            max_links_threshold=links_expected,
                        )
                else:
                    bt.logging.info(f"UID '{uid}' has no validator links.")
                    reward_event.reward = 0  # Handle case with no validator links
                reward_event.reward = min(reward_event.reward * apify_score, 1)
                reward_events.append(reward_event)
                grouped_val_score_responses.append(response_scores)

            zero_scores = {}
            non_zero_scores = {}

            for (index, response), uid_tensor, reward_e in zip(
                enumerate(responses), uids, reward_events
            ):
                uid = uid_tensor.item()
                if reward_e.reward == 0:
                    # score_explain = score_responses.get(str(uid), "")
                    zero_scores[uid] = reward_e.reward
                else:
                    non_zero_scores[uid] = reward_e.reward

            bt.logging.info(
                f"==================================Web Search Content Relevance scoring Zero Scores  ({len(zero_scores)} cases)=================================="
            )
            bt.logging.info(json.dumps(zero_scores))
            bt.logging.info(
                f"==================================Web Search Content Relevance scoring Non-Zero Scores ({len(non_zero_scores)} cases)=================================="
            )
            bt.logging.info(json.dumps(non_zero_scores))
            return reward_events, grouped_val_score_responses
        except Exception as e:
            error_message = f"Search Summary Relevance get_rewards: {str(e)}"
            tb_str = traceback.format_exception(type(e), e, e.__traceback__)
            bt.logging.error("\n".join(tb_str) + error_message)
            reward_events = []
            for response in responses:
                reward_event = BaseRewardEvent()
                reward_event.reward = 0
                reward_events.append(reward_event)
            return reward_events, {}
