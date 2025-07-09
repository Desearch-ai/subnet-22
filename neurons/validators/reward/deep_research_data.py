import asyncio
import traceback
import random
from typing import List, Dict, Tuple
import json
import bittensor as bt
from newspaper import Article

from neurons.validators.base_validator import AbstractNeuron

from .config import RewardModelType
from .reward import BaseRewardModel, BaseRewardEvent
from datura.protocol import DeepResearchSynapse, ReportItem
from neurons.validators.utils.prompt.deep_research.deep_research_data_relevance_prompt import (
    DeepResearchDataRelevancePrompt1,
)
from neurons.validators.apify.cheerio_scraper_actor import CheerioScraperActor
from neurons.validators.apify.utils import scrape_links_with_retries

RANDOM_SECTIONS_COUNT = 2
RANDOM_SECTION_LINKS_COUNT = 1


class DeepResearchDataRelevanceModel(BaseRewardModel):
    @property
    def name(self) -> str:
        return RewardModelType.deep_research_data_relevance.value

    def __init__(self, device: str, scoring_type: None, neuron: AbstractNeuron):
        super().__init__(neuron)
        self.device = device
        self.scoring_type = scoring_type
        self.relevance_prompt = DeepResearchDataRelevancePrompt1()

        self.is_default_normalization = False

    async def fetch_contents_batch(self, all_links: List[str]) -> Dict[str, str]:
        try:
            if not all_links:
                return {}

            links_with_metadata, _ = await scrape_links_with_retries(
                urls=all_links,
                scraper_actor_class=CheerioScraperActor,
                group_size=100,
                max_attempts=2,
            )

            # Create a mapping from URL to content
            url_to_content = {}

            for link_data in links_with_metadata:
                url = link_data.get("link", "")
                html = link_data.get("html_text", "")

                article = Article(url="")
                article.set_html(html)
                article.parse()

                url_to_content[url] = article.text

            return url_to_content
        except Exception as e:
            bt.logging.error(f"deep_research_data fetch_contents_batch error: {str(e)}")
            return {}

    async def check_section_data(
        self, section: ReportItem, url_to_content: Dict[str, str]
    ) -> float:
        try:
            # Get contents for this section's links from the pre-fetched dictionary
            result_texts = [url_to_content.get(link) for link in section.links]
            result_texts = [text for text in result_texts if text]

            response = await self.relevance_prompt.get_response(
                section.description, result_texts.__str__()
            )

            return self.relevance_prompt.extract_score(response) / 10, response
        except Exception as e:
            bt.logging.error(f"deep_research_data check_section_data error: {str(e)}")
            return 0.0, "Error"

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

            response_random_links = []

            for response in responses:
                sections = []

                # Add section and subsections of response to the list
                for section in response.report:
                    sections.append(section)
                    sections.extend(section.subsections)

                random_sections = random.sample(
                    response.report, k=min(RANDOM_SECTIONS_COUNT, len(response.report))
                )

                response.validator_items = random_sections

                # Pick random links from random sections that were selected
                random_links = set()

                for section in random_sections:
                    links = random.sample(
                        section.links,
                        k=min(RANDOM_SECTION_LINKS_COUNT, len(section.links)),
                    )

                    all_links.update(links)
                    random_links.update(links)

                response_random_links.append(list(random_links))

            # Step 2: Fetch all contents in a single batch
            url_to_content = await self.fetch_contents_batch(list(all_links))

            # Step 3: Process each response with the pre-fetched contents
            for response, random_links, uid_tensor in zip(
                responses, response_random_links, uids
            ):
                for link in random_links:
                    content = url_to_content.get(link, "")
                    response.validator_links[link] = content

                # If uid_tensor is a PyTorch or NumPy scalar, .item() extracts the integer
                uid = uid_tensor.item() if hasattr(uid_tensor, "item") else uid_tensor

                # Compute scores for each section using pre-fetched content
                scores = await asyncio.gather(
                    *[
                        self.check_section_data(section, url_to_content)
                        for section in response.validator_items
                    ]
                )

                scores = [score[0] for score in scores]
                final_score = sum(scores) / RANDOM_SECTIONS_COUNT if scores else 0.0

                bt.logging.info(
                    f"UID {uid}: deep research data relevance score => {final_score}"
                )

                # Create a reward event
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
                f"========== Deep Research Data Check Zero Scores ({len(zero_scores)} cases) =========="
            )
            bt.logging.info(json.dumps(zero_scores))
            bt.logging.info(
                f"======== Deep Research Data Check Non-Zero Scores ({len(non_zero_scores)} cases) ========"
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
