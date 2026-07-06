from desearch.protocol import (
    ScraperStreamingSynapse,
    TwitterSearchSynapse,
)
from neurons.validators.penalty.penalty import CheapPenaltyModel, PenaltyModelType

TWITTER_TOOL = "Twitter Search"
SEARCH_SUMMARY_TOOLS = ("Web Search",)
SEARCH_SUMMARY_FIELDS = ("search_results",)


class CountPenaltyModel(CheapPenaltyModel):
    """Penalize miners that return fewer results than the validator requested.

    Twitter uses ``count``. AI search checks per scoring group (Twitter; Web)
    — mirroring ``ScraperStreamingSynapse.get_search_results_by_tools``."""

    name = PenaltyModelType.count_penalty.value

    def penalty_for(self, response) -> float:
        if isinstance(response, TwitterSearchSynapse):
            requested = response.count
            got = len(response.results or [])
        elif isinstance(response, ScraperStreamingSynapse):
            return self._ai_search_shortfall(response)
        else:
            return 0.0

        if not requested or requested <= 0 or got >= requested:
            return 0.0
        return min(1 - got / requested, self.max_penalty)

    def _ai_search_shortfall(self, response: ScraperStreamingSynapse) -> float:
        """Worst per-group shortfall — pooled across tools in the same scoring group."""
        requested = response.count
        if not requested or requested <= 0:
            return 0.0

        tools = set(response.tools or [])
        group_totals = []

        if TWITTER_TOOL in tools:
            group_totals.append(len(response.miner_tweets or []))

        if any(t in tools for t in SEARCH_SUMMARY_TOOLS):
            group_totals.append(
                sum(
                    len(getattr(response, f, None) or []) for f in SEARCH_SUMMARY_FIELDS
                )
            )

        worst = 0.0
        for got in group_totals:
            if got < requested:
                worst = max(worst, 1 - got / requested)
        return min(worst, self.max_penalty)
