from desearch.protocol import (
    ScraperStreamingSynapse,
    TwitterSearchSynapse,
    WebSearchSynapse,
)
from neurons.validators.penalty.penalty import CheapPenaltyModel, PenaltyModelType

TWITTER_TOOL = "Twitter Search"
SEARCH_SUMMARY_TOOLS = ("Web Search", "Wikipedia Search", "Youtube Search", "ArXiv Search")
SEARCH_SUMMARY_FIELDS = (
    "search_results",
    "wikipedia_search_results",
    "youtube_search_results",
    "arxiv_search_results",
)
REDDIT_TOOL = "Reddit Search"
HACKER_NEWS_TOOL = "Hacker News Search"


class CountPenaltyModel(CheapPenaltyModel):
    """Penalize miners that return fewer results than the validator requested.

    Twitter uses ``count`` and Web uses ``num``. AI search checks per scoring
    group (Twitter; pooled web+wiki+yt+arxiv; Reddit; Hacker News) — mirroring
    ``ScraperStreamingSynapse.get_search_results_by_tools``."""

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
                sum(len(getattr(response, f, None) or []) for f in SEARCH_SUMMARY_FIELDS)
            )

        if REDDIT_TOOL in tools:
            group_totals.append(len(response.reddit_search_results or []))

        if HACKER_NEWS_TOOL in tools:
            group_totals.append(len(response.hacker_news_search_results or []))

        worst = 0.0
        for got in group_totals:
            if got < requested:
                worst = max(worst, 1 - got / requested)
        return min(worst, self.max_penalty)
