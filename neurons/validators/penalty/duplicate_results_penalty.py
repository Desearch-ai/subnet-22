from typing import List

import bittensor as bt
import numpy as np

from desearch.protocol import (
    ScraperStreamingSynapse,
    TwitterSearchSynapse,
    WebSearchSynapse,
)
from neurons.validators.base_validator import AbstractNeuron
from neurons.validators.penalty.penalty import BasePenaltyModel, PenaltyModelType
from neurons.validators.utils.response_checks import first_duplicate_id

MAX_PENALTY = 1.0


class DuplicateResultsPenaltyModel(BasePenaltyModel):
    """Penalize responses with duplicate result IDs / URLs. Catches miners
    padding their result count with copies of the same item."""

    is_deep = False

    def __init__(self, max_penalty: float = MAX_PENALTY, neuron: AbstractNeuron = None):
        super().__init__(max_penalty, neuron)

    @property
    def name(self) -> str:
        return PenaltyModelType.duplicate_results_penalty.value

    @staticmethod
    def _result_groups(response):
        """Return list of (items, key) pairs to check for duplicates."""
        if isinstance(response, TwitterSearchSynapse):
            return [(response.results or [], "id")]
        if isinstance(response, WebSearchSynapse):
            return [(response.results or [], "link")]
        if isinstance(response, ScraperStreamingSynapse):
            groups = []
            if response.miner_tweets:
                groups.append((response.miner_tweets, "id"))
            for field in (
                "search_results",
                "wikipedia_search_results",
                "youtube_search_results",
                "arxiv_search_results",
                "reddit_search_results",
                "hacker_news_search_results",
            ):
                groups.append((getattr(response, field, []) or [], "link"))
            return groups
        return []

    async def calculate_penalties(
        self,
        responses: List[bt.Synapse],
        additional_params=None,
    ) -> np.ndarray:
        penalties = np.zeros(len(responses), dtype=np.float32)
        for i, response in enumerate(responses):
            for items, key in self._result_groups(response):
                if first_duplicate_id(items, key=key) is not None:
                    penalties[i] = self.max_penalty
                    break
        return penalties
