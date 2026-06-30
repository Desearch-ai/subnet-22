from typing import List

import numpy as np

from desearch.protocol import ScraperStreamingSynapse
from neurons.validators.reward.config import RewardModelType, RewardScoringType
from neurons.validators.reward.reward import BaseRewardEvent, BaseRewardModel
from neurons.validators.reward.reward_llm import RewardLLM
from neurons.validators.reward.search_content_relevance import (
    WebSearchContentRelevanceModel,
    response_uses_web_tools,
)
from neurons.validators.reward.twitter_content_relevance import (
    TwitterContentRelevanceModel,
    response_uses_twitter_tool,
)


class ContentRelevanceRewardModel(BaseRewardModel):
    @property
    def name(self) -> str:
        return RewardModelType.content_relevance.value

    def __init__(self, llm_reward: RewardLLM, neuron):
        super().__init__(neuron)
        self.twitter = TwitterContentRelevanceModel(
            scoring_type=RewardScoringType.summary_relevance_score_template,
            llm_reward=llm_reward,
            neuron=neuron,
        )
        self.web = WebSearchContentRelevanceModel(
            scoring_type=RewardScoringType.search_relevance_score_template,
            llm_reward=llm_reward,
            neuron=neuron,
        )

    async def get_rewards(self, responses: List[ScraperStreamingSynapse], uids):
        uids = np.asarray(uids)
        events = [BaseRewardEvent(reward=0.0) for _ in responses]
        labels = [{} for _ in responses]

        twitter_idx, web_idx = [], []
        for i, response in enumerate(responses):
            if response_uses_twitter_tool(response):
                twitter_idx.append(i)
            elif response_uses_web_tools(response):
                web_idx.append(i)

        for model, idx in ((self.twitter, twitter_idx), (self.web, web_idx)):
            if not idx:
                continue
            sub_events, sub_labels = await model.get_rewards(
                [responses[i] for i in idx], uids[np.array(idx)]
            )
            for j, i in enumerate(idx):
                events[i] = sub_events[j]
                if isinstance(sub_labels, list) and j < len(sub_labels):
                    labels[i] = sub_labels[j]

        return events, labels
