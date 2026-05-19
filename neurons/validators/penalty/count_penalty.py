from desearch.protocol import (
    ScraperStreamingSynapse,
    TwitterSearchSynapse,
    WebSearchSynapse,
)
from neurons.validators.penalty.penalty import CheapPenaltyModel, PenaltyModelType
from neurons.validators.utils.response_checks import AI_ALL_RESULT_FIELDS


class CountPenaltyModel(CheapPenaltyModel):
    """Penalize miners that return fewer results than the validator requested.

    Twitter uses ``count`` and Web uses ``num``. AI search uses ``count`` as a
    per-source target and is checked against every populated result field — if
    the miner returned items for a source, that source must hit the count."""

    name = PenaltyModelType.count_penalty.value

    def penalty_for(self, response) -> float:
        if isinstance(response, TwitterSearchSynapse):
            requested = response.count
            got = len(response.results or [])
        elif isinstance(response, WebSearchSynapse):
            requested = response.num
            got = len(response.results or [])
        elif isinstance(response, ScraperStreamingSynapse):
            return self._ai_search_shortfall(response)
        else:
            return 0.0

        if not requested or requested <= 0 or got >= requested:
            return 0.0
        return min(1 - got / requested, self.max_penalty)

    def _ai_search_shortfall(self, response: ScraperStreamingSynapse) -> float:
        """Largest shortfall across populated result fields, in [0, 1]."""
        requested = response.count
        if not requested or requested <= 0:
            return 0.0
        worst = 0.0
        for field in AI_ALL_RESULT_FIELDS:
            items = getattr(response, field, None) or []
            if items and len(items) < requested:
                worst = max(worst, 1 - len(items) / requested)
        return min(worst, self.max_penalty)
