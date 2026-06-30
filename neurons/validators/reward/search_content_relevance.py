import time
import traceback
from typing import List

import bittensor as bt

from desearch.protocol import ScraperStreamingSynapse, ScraperTextRole
from neurons.validators.apify.body_fetch import get_body_fetcher
from neurons.validators.base_validator import AbstractNeuron
from neurons.validators.penalty.count_penalty import SEARCH_SUMMARY_TOOLS
from neurons.validators.reward.reward_llm import RewardLLM
from neurons.validators.utils.prompts import (
    BodyLinkRelevancePrompt,
    build_body_relevance_messages,
)
from neurons.validators.utils.response_checks import (
    normalize_source_url,
    parse_tweet_date,
    tweet_date_in_range,
)
from neurons.validators.utils.source_bodies import (
    cited_urls_normalized,
    highlights_in_order,
    sample_cited_and_uncited,
)

from .config import RewardModelType
from .reward import BaseRewardEvent, BaseRewardModel, log_reward_aggregates

WEB_TOOLS = frozenset(SEARCH_SUMMARY_TOOLS)
MAX_SAMPLED_LINKS = 3
MAX_CITED_SAMPLE = 2


def response_uses_web_tools(response: ScraperStreamingSynapse) -> bool:
    return bool(set(response.tools or []) & WEB_TOOLS)


def link_meets_evidence(miner_highlights, miner_text, fetched_body) -> bool:
    if not miner_highlights or not miner_text:
        return False
    if not highlights_in_order(miner_highlights, fetched_body):
        return False
    if not highlights_in_order(miner_highlights, miner_text):
        return False
    return True


class WebSearchContentRelevanceModel(BaseRewardModel):
    reward_model_name: str = "VMware/open-llama-7b-open-instruct"

    @property
    def name(self) -> str:
        return RewardModelType.search_content_relevance.value

    def __init__(
        self,
        scoring_type: None,
        llm_reward: RewardLLM,
        neuron: AbstractNeuron,
    ):
        super().__init__(neuron)
        self.reward_llm = llm_reward
        self.scoring_type = scoring_type

    async def llm_process_validator_links(self, response: ScraperStreamingSynapse):
        if not response_uses_web_tools(response) or not response.validator_links:
            return {}

        scoring_messages = []

        for validator_link in response.validator_links:
            url = validator_link.get("link")
            title = validator_link.get("title", "")
            body = validator_link.get("body", "")

            miner_highlights = validator_link.get("miner_highlights") or []
            miner_text = validator_link.get("miner_text") or ""

            if not link_meets_evidence(miner_highlights, miner_text, body):
                continue

            judged_body = "\n\n".join(miner_highlights)
            messages = build_body_relevance_messages(
                response.prompt, url, title, judged_body
            )
            if messages:
                scoring_messages.append({url: messages})

        score_responses = await self.reward_llm.llm_processing(scoring_messages)
        return score_responses

    async def scrape_links(self, urls):
        bodies_map = await get_body_fetcher().get_many(urls)

        fetched_links_with_metadata = [
            {
                "link": url,
                "title": body.get("title", ""),
                "body": body.get("text", ""),
                "published_date": body.get("published_date", ""),
                "author": body.get("author", ""),
            }
            for url, body in bodies_map.items()
            if url and body.get("text")
        ]
        fetched_urls = {link["link"] for link in fetched_links_with_metadata}
        non_fetched_links = [u for u in urls if u not in fetched_urls]

        return fetched_links_with_metadata, non_fetched_links

    def _miner_link_metadata(self, response):
        meta = {}
        for result in response.search_results or []:
            link = result.get("link")
            if not link:
                continue
            meta[normalize_source_url(link)] = {
                "highlights": result.get("highlights"),
                "text": result.get("text"),
            }
        return meta

    def _sample_cited_and_other(self, response, links_per_tool_group):
        summary = response.texts.get(ScraperTextRole.FINAL_SUMMARY.value, "")
        cited_norm = cited_urls_normalized(summary)

        flat = [link for group in links_per_tool_group.values() for link in group]
        picks = sample_cited_and_uncited(
            flat, cited_norm, MAX_CITED_SAMPLE, MAX_SAMPLED_LINKS
        )
        return picks, cited_norm

    async def process_links(self, responses: List[ScraperStreamingSynapse]):
        default_val_score_responses = [{} for _ in responses]

        start_time = time.time()

        all_links = []
        responses_random_links = [[] for _ in responses]
        responses_uncited_links = [[] for _ in responses]

        for response, random_links, uncited_links in zip(
            responses, responses_random_links, responses_uncited_links
        ):
            if not response_uses_web_tools(response):
                continue

            completion = self.get_successful_search_summary_completion(response)

            if not completion:
                continue

            _, links_per_tool_group = response.get_links_from_search_results()
            links, cited_norm = self._sample_cited_and_other(
                response, links_per_tool_group
            )
            uncited_links.extend(
                link for link in links if normalize_source_url(link) not in cited_norm
            )

            random_links.extend(links)
            all_links.extend(links)

        attempted_counts = [len(rl) for rl in responses_random_links]
        zero_uncited_unfetched = [0 for _ in responses]

        unique_links = list(set(all_links))

        if len(unique_links) == 0:
            bt.logging.info("No unique links found to process.")
            return default_val_score_responses, attempted_counts, zero_uncited_unfetched

        bt.logging.info(f"Fetching {len(unique_links)} unique web links.")

        links_with_metadata, non_fetched_links = await self.scrape_links(unique_links)

        if not links_with_metadata:
            bt.logging.info(
                "No validator web links were fetched. Returning empty score responses."
            )
            return default_val_score_responses, attempted_counts, zero_uncited_unfetched

        fetched_urls = {link.get("link") for link in links_with_metadata}
        uncited_unfetched_counts = [
            sum(1 for link in uncited if link not in fetched_urls)
            for uncited in responses_uncited_links
        ]

        for response, random_links in zip(responses, responses_random_links):
            miner_meta = self._miner_link_metadata(response)
            for link_with_metadata in links_with_metadata:
                url = link_with_metadata.get("link")

                if url in random_links:
                    meta = miner_meta.get(normalize_source_url(url), {})
                    response.validator_links.append(
                        {
                            **link_with_metadata,
                            "miner_highlights": meta.get("highlights") or [],
                            "miner_text": meta.get("text") or "",
                        }
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

        val_score_responses_list = await self.process_response_items_in_batches(
            responses=responses,
            batch_size=20,
            process_function=self.llm_process_validator_links,
        )

        return val_score_responses_list, attempted_counts, uncited_unfetched_counts

    def check_response_random_link(self, response: ScraperStreamingSynapse):
        try:
            if not response_uses_web_tools(response):
                return 0

            completion = self.get_successful_search_summary_completion(
                response=response
            )

            if not completion:
                return 0

            search_result_links, _ = response.get_links_from_search_results()
            validator_links = response.validator_links

            if not search_result_links or not validator_links:
                return 0

            if len(search_result_links) < 2:
                # at least miners should provide two search links
                return 0

            web_search_results = str(response.search_results)

            link_scores = []

            for val_link in validator_links:
                url = val_link.get("link")

                if not url:
                    link_scores.append(0)
                    continue

                has_evidence = link_meets_evidence(
                    val_link.get("miner_highlights") or [],
                    val_link.get("miner_text") or "",
                    val_link.get("body") or "",
                )
                link_scores.append(
                    1 if (url in web_search_results and has_evidence) else 0
                )

            if link_scores:
                return sum(link_scores) / len(link_scores)

            return 0
        except Exception as e:
            bt.logging.error(f"check_response_random_link: {str(e)}")
            return 0

    def _reward_from_link_scores(
        self, total_score: float, judged_count: int, uncited_unfetched: int
    ) -> float:
        denom = judged_count + uncited_unfetched
        if denom == 0:
            return 0.0
        return self.clamp_relevance_score(total_score / denom)

    def _web_date_blocks_link(self, response, val_link) -> bool:
        if not (response.start_date or response.end_date):
            return False
        validator_date = val_link.get("published_date") or ""
        if parse_tweet_date(validator_date) is None:
            return False
        return not tweet_date_in_range(
            validator_date, response.start_date, response.end_date
        )

    async def get_rewards(
        self, responses: List[ScraperStreamingSynapse], uids
    ) -> List[BaseRewardEvent]:
        try:
            completions: List[str] = self.get_successful_search_completions(responses)
            bt.logging.debug(
                f"WebSearchContentRelevanceModel | Calculating {len(completions)} rewards (typically < 1 sec/reward)."
            )

            (
                val_score_responses_list,
                attempted_counts,
                uncited_unfetched_counts,
            ) = await self.process_links(responses=responses)

            scores = [
                self.check_response_random_link(response) for response in responses
            ]

            reward_events = []
            scoring_prompt = BodyLinkRelevancePrompt()

            grouped_val_score_responses = []
            missing_validator_links = []

            for (
                apify_score,
                response,
                val_score_responses,
                attempted_count,
                uncited_unfetched,
                _,
            ) in zip(
                scores,
                responses,
                val_score_responses_list,
                attempted_counts,
                uncited_unfetched_counts,
                uids,
            ):
                reward_event = BaseRewardEvent()
                reward_event.reward = 0
                is_applicable = response_uses_web_tools(response)

                response_scores = {}
                total_score = 0

                for val_link in response.validator_links:
                    val_url = val_link.get("link")
                    if val_score_responses:
                        score_result = val_score_responses.get(val_url, None)
                        if score_result is not None:
                            score = scoring_prompt.extract_score(score_result)
                            relevance = scoring_prompt.contextual_relevance(
                                score_result
                            )
                            if self._web_date_blocks_link(response, val_link):
                                score = 0
                                relevance = "LOW"
                            total_score += score / 3.0
                            response_scores[val_url] = relevance

                judged_count = len(response_scores)
                if total_score > 0:
                    reward_event.reward = self._reward_from_link_scores(
                        total_score, judged_count, uncited_unfetched
                    )
                missing_validator_links.append(
                    1 if is_applicable and attempted_count == 0 else 0
                )

                reward_event.reward = min(reward_event.reward * apify_score, 1)
                reward_events.append(reward_event)
                grouped_val_score_responses.append(response_scores)

            log_reward_aggregates(
                name=self.name,
                uids=uids,
                scores=[e.reward for e in reward_events],
                extras={"missing_val_links": missing_validator_links},
            )
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
