from typing import Any, Callable, Iterable, Tuple

from desearch.protocol import (
    ScraperStreamingSynapse,
    TwitterSearchSynapse,
)
from desearch.utils import is_valid_tweet, is_valid_web_search_result
from neurons.validators.penalty.penalty import CheapPenaltyModel, PenaltyModelType
from neurons.validators.utils.response_checks import AI_SEARCH_RESULT_FIELDS


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


def _groups(response) -> Iterable[Tuple[list, Callable]]:
    if isinstance(response, TwitterSearchSynapse):
        yield response.results or [], _is_valid_tweet
        return
    if isinstance(response, ScraperStreamingSynapse):
        yield response.miner_tweets or [], _is_valid_tweet
        for field in AI_SEARCH_RESULT_FIELDS:
            yield getattr(response, field, []) or [], _is_valid_search_item


class ResultSchemaPenaltyModel(CheapPenaltyModel):
    """Penalty scales with the fraction of results that fail their protocol
    schema or have empty required content fields (id/text/url/created_at for
    tweets; title/link/snippet for search items)."""

    name = PenaltyModelType.result_schema_penalty.value

    def penalty_for(self, response) -> float:
        total = 0
        invalid = 0
        for items, validator in _groups(response):
            for item in items:
                total += 1
                if not validator(item):
                    invalid += 1
        if total == 0:
            return 0.0
        return min(invalid / total, self.max_penalty)
