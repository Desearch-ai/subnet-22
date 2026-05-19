from desearch.protocol import (
    ScraperStreamingSynapse,
    TwitterSearchSynapse,
)
from neurons.validators.penalty.penalty import CheapPenaltyModel, PenaltyModelType
from neurons.validators.utils.response_checks import tweet_date_in_range


class DateRangePenaltyModel(CheapPenaltyModel):
    """Penalize responses whose tweets fall outside the requested
    [start_date, end_date]. Pure code — checks the miner's claimed
    ``created_at``; the deep model verifies the claim against Apify."""

    name = PenaltyModelType.date_range_penalty.value

    def penalty_for(self, response) -> float:
        tweets, start_date, end_date = self._tweets_and_bounds(response)
        if not tweets or (not start_date and not end_date):
            return 0.0

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
            return 0.0
        return min(out_of_range / checked, self.max_penalty)

    @staticmethod
    def _tweets_and_bounds(response):
        if isinstance(response, TwitterSearchSynapse):
            return response.results or [], response.start_date, response.end_date
        if isinstance(response, ScraperStreamingSynapse):
            return response.miner_tweets or [], response.start_date, response.end_date
        return [], None, None
