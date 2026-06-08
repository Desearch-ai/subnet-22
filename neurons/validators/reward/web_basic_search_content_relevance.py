import html
import random
import re
import time
import traceback
from typing import Dict, List, Tuple

import bittensor as bt

from desearch.protocol import WebSearchSynapse, WebSearchValidatorResult
from desearch.utils import is_valid_web_search_result
from neurons.validators.apify.scrapingdog_scraper import (
    scrape_links_with_retries,
)
from neurons.validators.base_validator import AbstractNeuron
from neurons.validators.reward.reward_llm import RewardLLM
from neurons.validators.utils.prompts import WebSearchRelevancePrompt
from neurons.validators.utils.web_query_operators import parse_web_query

from .config import RewardModelType
from .reward import BaseRewardEvent, BaseRewardModel, log_reward_aggregates

WEB_LINK_SCRAPE_AMOUNT = 1

MIN_SNIPPET_CHARS = 40
MIN_SNIPPET_DISTINCT_TOKENS = 5
SNIPPET_BIGRAM_THRESHOLD = 0.5

_TOKEN_RE = re.compile(r"\w+")
_STOPWORDS = frozenset(
    "a an and are as at be but by for from has have how in is it its more most of on "
    "or that the these this those to was were what when where which who will with you "
    "your we our they i".split()
)


class WebBasicSearchContentRelevanceModel(BaseRewardModel):
    @property
    def name(self) -> str:
        return RewardModelType.web_basic_search_content_relevance.value

    def __init__(
        self, scoring_type: None, neuron: AbstractNeuron, llm_reward: RewardLLM
    ):
        super().__init__(neuron)
        self.scoring_type = scoring_type
        self.reward_llm = llm_reward

    def normalize_html_content(self, content: str) -> str:
        if content is None:
            return ""

        normalized_content = re.sub(
            r"\s+", " ", content.replace("\n", " ").replace("\r", " ").strip()
        )
        return html.unescape(normalized_content).lower()

    @staticmethod
    def _tokens(text: str) -> List[str]:
        return _TOKEN_RE.findall(text.lower()) if text else []

    @classmethod
    def _is_substantive_snippet(cls, snippet: str) -> bool:
        if not snippet or len(snippet.strip()) < MIN_SNIPPET_CHARS:
            return False
        distinct_content = {t for t in cls._tokens(snippet) if t not in _STOPWORDS}
        return len(distinct_content) >= MIN_SNIPPET_DISTINCT_TOKENS

    def _snippet_bigram_overlap(
        self, snippet: str, validator_item: WebSearchValidatorResult
    ) -> float:
        tokens = self._tokens(self.normalize_html_content(snippet))
        snippet_bigrams = set(zip(tokens, tokens[1:]))
        if not snippet_bigrams:
            return 0.0

        best = 0.0
        for target in (
            validator_item.html_content,
            validator_item.html_text,
            validator_item.snippet,
        ):
            target_tokens = self._tokens(self.normalize_html_content(target))
            target_bigrams = set(zip(target_tokens, target_tokens[1:]))
            if not target_bigrams:
                continue
            overlap = sum(1 for b in snippet_bigrams if b in target_bigrams) / len(
                snippet_bigrams
            )
            best = max(best, overlap)
        return best

    def _snippet_verified(
        self, snippet: str, validator_item: WebSearchValidatorResult
    ) -> bool:
        return (
            self._is_substantive_snippet(snippet)
            and self._snippet_bigram_overlap(snippet, validator_item)
            >= SNIPPET_BIGRAM_THRESHOLD
        )

    async def scrape_links(self, urls):
        (
            fetched_links_with_metadata,
            non_fetched_links,
        ) = await scrape_links_with_retries(
            urls=urls,
            max_attempts=2,
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
                    min(WEB_LINK_SCRAPE_AMOUNT, len(urls)),
                )

                random_links.extend(sample_links)
                all_links.extend(sample_links)

        unique_links = list(set(all_links))

        if len(unique_links) == 0:
            bt.logging.info("No unique links found to process.")
            return default_val_score_responses

        bt.logging.info(f"Fetching {len(unique_links)} unique web links.")

        links_with_metadata, non_fetched_links = await self.scrape_links(unique_links)

        if not links_with_metadata:
            bt.logging.info(
                "No validator web links were fetched. Returning empty score responses."
            )
            return default_val_score_responses

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
            f"Validator fetched web links count: {len(links_with_metadata)}"
        )

        bt.logging.info(
            f"Web links not fetched amount: {len(non_fetched_links)}; List: {non_fetched_links}"
        )
        if len(non_fetched_links):
            bt.logging.info(
                f"Unique Web Links Amount: {len(unique_links)}; List: {unique_links};"
            )

        return default_val_score_responses

    async def llm_process_validator_links(self, response: WebSearchSynapse):
        if not response.validator_links:
            return {}

        scoring_prompt = WebSearchRelevancePrompt()
        query_text = parse_web_query(response.query).text or response.query
        scoring_messages = []

        for validator_link in response.validator_links:
            content = f"Title: {validator_link.title or ''}, Description: {validator_link.snippet or ''}"
            scoring_messages.append(
                {
                    validator_link.link: [
                        {
                            "role": "system",
                            "content": scoring_prompt.get_system_message(),
                        },
                        {
                            "role": "user",
                            "content": scoring_prompt.text(query_text, content),
                        },
                    ]
                }
            )

        return await self.reward_llm.llm_processing(scoring_messages)

    @staticmethod
    def _normalize_title_for_match(title: str) -> str:
        if not title:
            return ""
        t = title.lower()
        t = re.sub(r"\s+[-–—|]\s+[^-–—|]{1,50}\s*$", "", t)
        t = re.sub(r"\br/", "", t)
        return re.sub(r"\s+", " ", t).strip(" .")

    def check_title(self, miner_title, validator_title):
        miner_norm = self._normalize_title_for_match(miner_title)
        validator_norm = self._normalize_title_for_match(validator_title)

        if not miner_norm or not validator_norm:
            return False

        return miner_norm in validator_norm or validator_norm in miner_norm

    def check_response_random_link(
        self, response: WebSearchSynapse, relevance_scores: Dict[str, float]
    ) -> float:
        try:
            miner_results = response.results
            validator_links = response.validator_links
            operators = parse_web_query(response.query)

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

                if not operators.host_allowed(validator_item.link):
                    scores.append(0)
                    continue

                if not self.check_title(miner_item.get("title"), validator_item.title):
                    scores.append(0)
                    continue

                if miner_item.get("link") != validator_item.link:
                    scores.append(0)
                    continue

                if not self._snippet_verified(
                    miner_item.get("snippet"), validator_item
                ):
                    scores.append(0)
                    continue

                relevance = relevance_scores.get(validator_item.link)

                if relevance is None:
                    continue

                scores.append(relevance)

            return sum(scores) / len(scores) if scores else 0.0
        except Exception as e:
            bt.logging.error(f"check_response_random_link error: {str(e)}")
            return 0.0

    async def get_rewards(
        self, responses: List[WebSearchSynapse], uids: List[int]
    ) -> Tuple[List[BaseRewardEvent], Dict[int, float]]:
        try:
            await self.process_links(responses=responses)

            val_score_responses_list = await self.process_response_items_in_batches(
                responses=responses,
                batch_size=20,
                process_function=self.llm_process_validator_links,
            )

            scoring_prompt = WebSearchRelevancePrompt()

            reward_events = []
            grouped_val_score_responses = {}

            for response, val_score_responses, uid_tensor in zip(
                responses, val_score_responses_list, uids
            ):
                uid = uid_tensor.item() if hasattr(uid_tensor, "item") else uid_tensor

                relevance_scores = {
                    url: scoring_prompt.extract_score(text)
                    for url, text in (val_score_responses or {}).items()
                    if text
                }

                final_score = self.check_response_random_link(
                    response, relevance_scores
                )

                reward_event = BaseRewardEvent()
                reward_event.reward = final_score
                reward_events.append(reward_event)

                grouped_val_score_responses[uid] = final_score

            log_reward_aggregates(
                name=self.name,
                uids=uids,
                scores=[e.reward for e in reward_events],
            )

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
