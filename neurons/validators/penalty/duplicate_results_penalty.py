from desearch.protocol import (
    ScraperStreamingSynapse,
    TwitterSearchSynapse,
    WebSearchSynapse,
)
from neurons.validators.penalty.penalty import CheapPenaltyModel, PenaltyModelType
from neurons.validators.utils.response_checks import (
    AI_SEARCH_RESULT_FIELDS,
    first_duplicate_id,
)


def _result_groups(response):
    """Yield ``(items, dedup_key)`` for every result list to check."""
    if isinstance(response, TwitterSearchSynapse):
        yield response.results or [], "id"
    elif isinstance(response, WebSearchSynapse):
        yield response.results or [], "link"
    elif isinstance(response, ScraperStreamingSynapse):
        if response.miner_tweets:
            yield response.miner_tweets, "id"
        for field in AI_SEARCH_RESULT_FIELDS:
            yield getattr(response, field, []) or [], "link"


class DuplicateResultsPenaltyModel(CheapPenaltyModel):
    """Penalize responses with duplicate result IDs / URLs. Catches miners
    padding their result count with copies of the same item."""

    name = PenaltyModelType.duplicate_results_penalty.value

    def penalty_for(self, response) -> float:
        for items, key in _result_groups(response):
            if first_duplicate_id(items, key=key) is not None:
                return self.max_penalty
        return 0.0
