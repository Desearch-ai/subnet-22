from typing import Type
import bittensor as bt
from pydantic import BaseModel, Field
from starlette.types import Send
from desearch.tools.base import BaseTool
from desearch.dataset.date_filters import get_specified_date_filter, DateFilterType
from neurons.validators.apify.twitter_scraper_actor import TwitterScraperActor


class TwitterSearchToolSchema(BaseModel):
    query: str = Field(
        ...,
        description="The search query for tweets.",
    )


class TwitterSearchTool(BaseTool):
    """Tool that gets tweets from Twitter."""

    name = "Twitter Search"

    slug = "get_tweets"

    description = "Get tweets for a given query."

    args_schema: Type[TwitterSearchToolSchema] = TwitterSearchToolSchema

    tool_id = "e831f03f-3282-4d5c-ae01-d7274515194d"

    def _run():
        pass

    async def _arun(
        self,
        query: str,  # run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Tweet message and return."""
        date_filter = (
            self.tool_manager.date_filter
            if self.tool_manager
            else get_specified_date_filter(DateFilterType.PAST_WEEK)
        )
        start_date = date_filter.start_date.date()
        end_date = date_filter.end_date.date()

        max_items = 10

        if self.tool_manager and self.tool_manager.synapse:
            max_items = self.tool_manager.synapse.count or 10

        client = TwitterScraperActor()
        tweets = await client.get_tweets_advanced(
            start=start_date,
            end=end_date,
            maxItems=max_items,
            searchTerms=[query],
        )

        if isinstance(tweets, dict) and tweets.get("error"):
            bt.logging.error(f"Twitter search failed: {tweets['error']}")
            return []

        return [
            tweet.model_dump() if hasattr(tweet, "model_dump") else tweet
            for tweet in tweets
        ]

    async def send_event(self, send: Send, response_streamer, data):
        if not data:
            return

        await response_streamer.send_event("tweets", data)

        if data:
            bt.logging.info(f"Tweet data sent. Number of tweets: {len(data)}")
