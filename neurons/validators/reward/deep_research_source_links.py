import asyncio
import traceback
import random
from typing import List, Dict, Tuple
import json
import bittensor as bt
from .config import RewardModelType
from .reward import BaseRewardModel, BaseRewardEvent
from datura.protocol import DeepResearchSynapse, ReportItem
from neurons.validators.utils.prompt.deep_research.deep_research_source_links_relevance_prompt import (
    DeepResearchSourceLinksRelevancePrompt,
)
from datura.synapse import collect_responses
from neurons.validators.apify.cheerio_scraper_actor import CheerioScraperActor
from neurons.validators.apify.utils import scrape_links_with_retries

RANDOM_SECTIONS_COUNT = 1
RANDOM_LINKS_COUNT = 3


class DeepResearchSourceLinksRelevanceModel(BaseRewardModel):
    @property
    def name(self) -> str:
        return RewardModelType.deep_research_source_links_relevance.value

    def __init__(self, device: str, scoring_type: None):
        super().__init__()
        self.device = device
        self.scoring_type = scoring_type
        self.relevance_prompt = DeepResearchSourceLinksRelevancePrompt()
        self.cheerio_scraper_actor = CheerioScraperActor()

        self.is_default_normalization = False

    def get_section_links(self, section: ReportItem):
        links = section.links
        return links

    async def fetch_contents_batch(self, all_links: List[str]) -> Dict[str, str]:
        try:
            if not all_links:
                return {}

            links_with_metadata = await scrape_links_with_retries(
                urls=all_links,
                scraper_actor_class=CheerioScraperActor,
                group_size=100,
                max_attempts=2,
            )

            # Create a mapping from URL to content
            url_to_content = {}
            for link_data in links_with_metadata:
                url = link_data.get("url", "")
                content = link_data.get("html_text", "")
                url_to_content[url] = content

            return url_to_content
        except Exception as e:
            bt.logging.error(
                f"deep_research_source_links fetch_contents_batch error: {str(e)}"
            )
            return {}

    async def check_section_link_content(self, content: str, prompt: str):
        try:
            response = await self.relevance_prompt.get_response(content, prompt)
            score = self.relevance_prompt.extract_score(response) / 10

            return score, response
        except Exception as e:
            bt.logging.error(
                f"deep_research_source_links check_section_link_content error: {str(e)}"
            )
            return 0.0, "Error"

    async def check_section_links(
        self, section: ReportItem, prompt: str, url_to_content: Dict[str, str]
    ):
        try:
            links = self.get_section_links(section)

            if len(links) == 0:
                return 1.0, "Doesn't have any links"

            # Select random links to check
            if len(links) > RANDOM_LINKS_COUNT:
                selected_links = random.sample(links, k=RANDOM_LINKS_COUNT)
            else:
                selected_links = links

            # Get the content for each link from the pre-fetched dictionary
            selected_contents = [
                url_to_content.get(link, "") for link in selected_links
            ]

            # Check each content
            scores = await collect_responses(
                [
                    self.check_section_link_content(content, prompt)
                    for content in selected_contents
                ]
            )
            llm_responses = [score[1] for score in scores]
            scores = [score[0] for score in scores]

            return sum(scores) / len(selected_links) if scores else 0.0, llm_responses
        except Exception as e:
            bt.logging.error(
                f"deep_research_source_links check_section_links error: {str(e)}"
            )
            return 0.0, "Error"

    async def check_response(
        self, synapse: DeepResearchSynapse, url_to_content: Dict[str, str]
    ) -> float:
        try:
            all_sections = [item for item in synapse.report if len(item.links) > 0]
            for report in synapse.report:
                all_sections.extend(
                    [item for item in report.subsections if len(item.links) > 0]
                )

            if not all_sections:
                return 0.0

            if len(all_sections) > RANDOM_SECTIONS_COUNT:
                sections = random.sample(all_sections, k=RANDOM_SECTIONS_COUNT)
            else:
                sections = all_sections

            scores = await collect_responses(
                [
                    self.check_section_links(section, synapse.prompt, url_to_content)
                    for section in sections
                ]
            )

            scores = [score[0] for score in scores]

            return sum(scores) / len(scores) if scores else 0.0

        except Exception as e:
            bt.logging.error(
                f"deep_research_source_links check_response error: {str(e)}"
            )
            return 0.0

    async def get_rewards(
        self, responses: List[DeepResearchSynapse], uids: List[int]
    ) -> Tuple[List[BaseRewardEvent], Dict[int, float]]:
        try:
            reward_events = []
            zero_scores = {}
            non_zero_scores = {}
            grouped_val_score_responses = {}

            # Step 1: Collect all links from all responses and sections
            all_links = set()

            for response in responses:
                all_sections = [item for item in response.report if len(item.links) > 0]
                for report in response.report:
                    all_sections.extend(
                        [item for item in report.subsections if len(item.links) > 0]
                    )

                # Collect links from all sections that might be selected
                for section in all_sections:
                    links = self.get_section_links(section)
                    if len(links) > RANDOM_LINKS_COUNT:
                        # Add a sample of links that might be selected during evaluation
                        all_links.update(random.sample(links, k=RANDOM_LINKS_COUNT))
                    else:
                        all_links.update(links)

            # Step 2: Fetch all contents in a single batch
            url_to_content = await self.fetch_contents_batch(list(all_links))

            # Step 3: Process each response with the pre-fetched contents
            for response, uid_tensor in zip(responses, uids):
                # If uid_tensor is a PyTorch or NumPy scalar, .item() extracts the integer
                uid = uid_tensor.item() if hasattr(uid_tensor, "item") else uid_tensor

                final_score = await self.check_response(response, url_to_content)

                bt.logging.info(
                    f"UID {uid}: deep research source links relevance score => {final_score}"
                )

                # Step 4: create a reward event
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

            # Step 5: Log zero vs. non-zero
            bt.logging.info(
                f"========== Deep Research Source Links Relevance Check Zero Scores ({len(zero_scores)} cases) =========="
            )
            bt.logging.info(json.dumps(zero_scores))
            bt.logging.info(
                f"======== Deep Research Source Links Relevance Check Non-Zero Scores ({len(non_zero_scores)} cases) ========"
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
