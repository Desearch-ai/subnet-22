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

MAX_PENALTY = 1.0

AI_SEARCH_RESULT_FIELDS = (
    "miner_tweets",
    "search_results",
    "wikipedia_search_results",
    "youtube_search_results",
    "arxiv_search_results",
    "reddit_search_results",
    "hacker_news_search_results",
)


class CountPenaltyModel(BasePenaltyModel):
    """Penalize miners that return fewer results than the validator requested.

    Twitter uses ``count`` and Web uses ``num``. AI search uses ``count`` as
    a per-source target and is checked against every populated result field —
    if the miner returned items for a source, that source must hit the count."""

    is_deep = False

    def __init__(self, max_penalty: float = MAX_PENALTY, neuron: AbstractNeuron = None):
        super().__init__(max_penalty, neuron)

    @property
    def name(self) -> str:
        return PenaltyModelType.count_penalty.value

    @staticmethod
    def _ai_search_shortfall(response: ScraperStreamingSynapse, requested: int) -> float:
        """Largest shortfall across populated result fields, expressed as a
        ratio in [0, 1]. Empty fields are skipped (the miner didn't claim that
        source)."""
        worst = 0.0
        for field in AI_SEARCH_RESULT_FIELDS:
            items = getattr(response, field, None) or []
            if not items:
                continue
            if len(items) < requested:
                worst = max(worst, 1 - len(items) / requested)
        return worst

    async def calculate_penalties(
        self,
        responses: List[bt.Synapse],
        additional_params=None,
    ) -> np.ndarray:
        penalties = np.zeros(len(responses), dtype=np.float32)

        for i, response in enumerate(responses):
            if isinstance(response, TwitterSearchSynapse):
                requested = response.count
                if not requested or requested <= 0:
                    continue
                got = len(response.results or [])
                penalties[i] = max(0.0, 1 - got / requested) if got < requested else 0.0
            elif isinstance(response, WebSearchSynapse):
                requested = response.num
                if not requested or requested <= 0:
                    continue
                got = len(response.results or [])
                penalties[i] = max(0.0, 1 - got / requested) if got < requested else 0.0
            elif isinstance(response, ScraperStreamingSynapse):
                requested = response.count
                if not requested or requested <= 0:
                    continue
                penalties[i] = self._ai_search_shortfall(response, requested)
            bt.logging.debug(f"Response index {i} has penalty {penalties[i]}")

        return penalties
