import asyncio
import traceback
from typing import List, Dict, Tuple
import json
import bittensor as bt

from neurons.validators.base_validator import AbstractNeuron
from .config import RewardModelType
from .reward import BaseRewardModel, BaseRewardEvent
from desearch.protocol import DeepResearchSynapse, ReportItem
from neurons.validators.utils.prompt.deep_research.deep_research_source_links_relevance_prompt import (
    DeepResearchSourceLinksRelevancePrompt,
)
from neurons.validators.apify.cheerio_scraper_actor import CheerioScraperActor
from neurons.validators.apify.utils import scrape_links_with_retries
from .deep_research_data import RANDOM_SECTION_LINKS_COUNT, RANDOM_SECTIONS_COUNT


class DeepResearchSourceLinksRelevanceModel(BaseRewardModel):
    @property
    def name(self) -> str:
        return RewardModelType.deep_research_source_links_relevance.value

    def __init__(self, device: str, scoring_type: None, neuron: AbstractNeuron):
        super().__init__(neuron)
        self.device = device
        self.scoring_type = scoring_type
        self.relevance_prompt = DeepResearchSourceLinksRelevancePrompt()
        self.cheerio_scraper_actor = CheerioScraperActor()

        self.is_default_normalization = False

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
            links = section.links

            if not links:
                return 0.0, "Doesn't have any links"

            # Get the content for each link from the pre-fetched dictionary
            selected_contents = [url_to_content.get(link, "") for link in links]

            # Check each content
            scores = await asyncio.gather(
                *[
                    self.check_section_link_content(content, prompt)
                    for content in selected_contents
                ]
            )

            llm_responses = [score[1] for score in scores]
            scores = [score[0] for score in scores]

            return (
                sum(scores) / RANDOM_SECTION_LINKS_COUNT if scores else 0.0
            ), llm_responses
        except Exception as e:
            bt.logging.error(
                f"deep_research_source_links check_section_links error: {str(e)}"
            )
            return 0.0, "Error"

    async def check_response(self, synapse: DeepResearchSynapse) -> float:
        try:
            sections = synapse.validator_items

            if not sections:
                return 0.0

            scores = await asyncio.gather(
                *[
                    self.check_section_links(
                        section, synapse.prompt, synapse.validator_links
                    )
                    for section in sections
                ]
            )

            scores = [score[0] for score in scores]

            return sum(scores) / RANDOM_SECTIONS_COUNT if scores else 0.0
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

            # Step 1: Process each response with the pre-fetched contents
            for response, uid_tensor in zip(responses, uids):
                # If uid_tensor is a PyTorch or NumPy scalar, .item() extracts the integer
                uid = uid_tensor.item() if hasattr(uid_tensor, "item") else uid_tensor

                final_score = await self.check_response(response)

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
