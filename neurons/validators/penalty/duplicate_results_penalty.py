from desearch.protocol import (
    ScraperStreamingSynapse,
    TwitterSearchSynapse,
)
from desearch.utils import format_text_for_match
from neurons.validators.penalty.penalty import CheapPenaltyModel, PenaltyModelType
from neurons.validators.utils.response_checks import (
    AI_SEARCH_RESULT_FIELDS,
    first_duplicate_id,
    source_key,
)

_URL_KEYS = frozenset({"link", "url"})


def _result_groups(response):
    """Yield ``(items, dedup_keys, check_text)`` for every result list to check."""
    if isinstance(response, TwitterSearchSynapse):
        yield response.results or [], ("id", "url"), True
    elif isinstance(response, ScraperStreamingSynapse):
        if response.miner_tweets:
            yield response.miner_tweets, ("id", "url"), True
        for field in AI_SEARCH_RESULT_FIELDS:
            yield getattr(response, field, []) or [], ("link", "url"), False


def _has_duplicate_text(items) -> bool:
    seen: set[str] = set()
    for item in items or []:
        text = item.get("text") if isinstance(item, dict) else getattr(item, "text", "")
        normalized = format_text_for_match(text or "").lower()
        if not normalized:
            continue
        if normalized in seen:
            return True
        seen.add(normalized)
    return False


class DuplicateResultsPenaltyModel(CheapPenaltyModel):
    """Penalize responses with duplicate result IDs / URLs. Catches miners
    padding their result count with copies of the same item."""

    name = PenaltyModelType.duplicate_results_penalty.value

    def penalty_for(self, response) -> float:
        for items, keys, check_text in _result_groups(response):
            for key in keys:
                normalize = source_key if key in _URL_KEYS else None
                if first_duplicate_id(items, key=key, normalize=normalize) is not None:
                    return self.max_penalty
            if check_text and _has_duplicate_text(items):
                return self.max_penalty
        return 0.0
