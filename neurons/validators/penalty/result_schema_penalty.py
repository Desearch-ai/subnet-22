from typing import Any, Callable, Iterable, List, Tuple

import bittensor as bt
import numpy as np

from desearch.protocol import (
    ScraperStreamingSynapse,
    TwitterSearchSynapse,
    WebSearchSynapse,
)
from desearch.utils import is_valid_tweet, is_valid_web_search_result
from neurons.validators.base_validator import AbstractNeuron
from neurons.validators.penalty.penalty import BasePenaltyModel, PenaltyModelType

MAX_PENALTY = 1.0

AI_SEARCH_RESULT_FIELDS = (
    "search_results",
    "wikipedia_search_results",
    "youtube_search_results",
    "arxiv_search_results",
    "reddit_search_results",
    "hacker_news_search_results",
)


def _is_valid_tweet(item: Any) -> bool:
    if not isinstance(item, dict) or not is_valid_tweet(item):
        return False
    for field in ("id", "text", "url", "created_at"):
        if not item.get(field):
            return False
    return True


def _is_valid_search_item(item: Any) -> bool:
    if isinstance(item, dict):
        if not is_valid_web_search_result(item):
            return False
        title, link, snippet = item.get("title"), item.get("link"), item.get("snippet")
    else:
        title = getattr(item, "title", None)
        link = getattr(item, "link", None)
        snippet = getattr(item, "snippet", None)
    return all((title, link, snippet))


class ResultSchemaPenaltyModel(BasePenaltyModel):
    """Penalty scales with the fraction of results that fail their protocol
    schema or have empty required content fields (id/text/url/created_at for
    tweets; title/link/snippet for search items)."""

    is_deep = False

    def __init__(self, max_penalty: float = MAX_PENALTY, neuron: AbstractNeuron = None):
        super().__init__(max_penalty, neuron)

    @property
    def name(self) -> str:
        return PenaltyModelType.result_schema_penalty.value

    @staticmethod
    def _groups(response) -> Iterable[Tuple[list, Callable]]:
        if isinstance(response, TwitterSearchSynapse):
            yield response.results or [], _is_valid_tweet
            return
        if isinstance(response, WebSearchSynapse):
            yield response.results or [], _is_valid_search_item
            return
        if isinstance(response, ScraperStreamingSynapse):
            yield response.miner_tweets or [], _is_valid_tweet
            for field in AI_SEARCH_RESULT_FIELDS:
                yield getattr(response, field, []) or [], _is_valid_search_item

    async def calculate_penalties(
        self,
        responses: List[bt.Synapse],
        additional_params=None,
    ) -> np.ndarray:
        penalties = np.zeros(len(responses), dtype=np.float32)
        for i, response in enumerate(responses):
            total = 0
            invalid = 0
            for items, validator in self._groups(response):
                for item in items:
                    total += 1
                    if not validator(item):
                        invalid += 1
            if total > 0:
                penalties[i] = min(invalid / total, self.max_penalty)
        return penalties
