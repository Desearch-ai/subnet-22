from typing import List

import bittensor as bt
import numpy as np

from desearch.protocol import (
    ScraperStreamingSynapse,
    TwitterSearchSynapse,
)
from neurons.validators.base_validator import AbstractNeuron
from neurons.validators.penalty.penalty import BasePenaltyModel, PenaltyModelType
from neurons.validators.utils.response_checks import tweet_date_in_range

MAX_PENALTY = 1.0


class DateRangePenaltyModel(BasePenaltyModel):
    """Penalize responses whose tweets fall outside the requested
    [start_date, end_date]. Pure code — checks the miner's claimed
    ``created_at``; the deep model verifies the claim against Apify."""

    is_deep = False

    def __init__(self, max_penalty: float = MAX_PENALTY, neuron: AbstractNeuron = None):
        super().__init__(max_penalty, neuron)

    @property
    def name(self) -> str:
        return PenaltyModelType.date_range_penalty.value

    @staticmethod
    def _tweets_and_bounds(response):
        if isinstance(response, TwitterSearchSynapse):
            return response.results or [], response.start_date, response.end_date
        if isinstance(response, ScraperStreamingSynapse):
            return response.miner_tweets or [], response.start_date, response.end_date
        return [], None, None

    async def calculate_penalties(
        self,
        responses: List[bt.Synapse],
        additional_params=None,
    ) -> np.ndarray:
        penalties = np.zeros(len(responses), dtype=np.float32)

        for i, response in enumerate(responses):
            tweets, start_date, end_date = self._tweets_and_bounds(response)
            if not tweets or (not start_date and not end_date):
                continue

            checked = 0
            out_of_range = 0
            for tweet in tweets:
                created = tweet.get("created_at") if isinstance(tweet, dict) else None
                if not created:
                    continue
                checked += 1
                if not tweet_date_in_range(created, start_date, end_date):
                    out_of_range += 1

            if checked == 0:
                continue
            penalties[i] = min(out_of_range / checked, self.max_penalty)

        return penalties
