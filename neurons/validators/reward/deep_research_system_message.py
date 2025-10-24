import traceback
from typing import List, Dict, Tuple
import json
import bittensor as bt

from neurons.validators.base_validator import AbstractNeuron
from .config import RewardModelType
from .reward import BaseRewardModel, BaseRewardEvent
from desearch.protocol import DeepResearchSynapse, ReportItem
from neurons.validators.utils.prompt.deep_research.deep_research_system_message_relevance_prompt import (
    DeepResearchSystemMessageRelevancePrompt,
)
from neurons.validators.apify.web_scraper_actor import WebScraperActor


class DeepResearchSystemMessageRelevanceModel(BaseRewardModel):
    @property
    def name(self) -> str:
        return RewardModelType.deep_research_system_message_relevance.value

    def __init__(self, device: str, scoring_type: None, neuron: AbstractNeuron):
        super().__init__(neuron)
        self.device = device
        self.scoring_type = scoring_type
        self.relevance_prompt = DeepResearchSystemMessageRelevancePrompt()
        self.web_scraper_actor = WebScraperActor()

        self.is_default_normalization = False

    async def check_response(self, synapse: DeepResearchSynapse) -> float:
        try:
            if not synapse.system_message:
                return 1.0, ""

            response = await self.relevance_prompt.get_response(
                synapse.to_xml_report(), synapse.system_message
            )

            return self.relevance_prompt.extract_score(response) / 10, response
        except Exception as e:
            bt.logging.error(
                f"deep_research_system_message check_response error: {str(e)}"
            )
            return 0.0, "Error"

    async def get_rewards(
        self, responses: List[DeepResearchSynapse], uids: List[int]
    ) -> Tuple[List[BaseRewardEvent], Dict[int, float]]:
        try:

            reward_events = []
            zero_scores = {}
            non_zero_scores = {}
            grouped_val_score_responses = {}

            # Step 2: for each response, compute a final score
            for response, uid_tensor in zip(responses, uids):
                # If uid_tensor is a PyTorch or NumPy scalar, .item() extracts the integer
                uid = uid_tensor.item() if hasattr(uid_tensor, "item") else uid_tensor

                final_score, _explanation = await self.check_response(response)

                bt.logging.info(
                    f"UID {uid}: deep research system message relevance score => {final_score}"
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
                f"========== Deep Research System Message Check Zero Scores ({len(zero_scores)} cases) =========="
            )
            bt.logging.info(json.dumps(zero_scores))
            bt.logging.info(
                f"======== Deep Research System Message Check Non-Zero Scores ({len(non_zero_scores)} cases) ========"
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
