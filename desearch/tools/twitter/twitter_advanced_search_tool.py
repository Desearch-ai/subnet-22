import json
import time
from typing import Type
import bittensor as bt
from neurons.validators.apify.twitter_scraper_actor import TwitterScraperActor
from pydantic import BaseModel, Field
from starlette.types import Send
from desearch.tools.base import BaseTool
from desearch.services.twitter_prompt_analyzer import TwitterPromptAnalyzer
from desearch.dataset.date_filters import get_specified_date_filter, DateFilterType


class TwitterAdvancedSearchToolSchema(BaseModel):
    query: str = Field(
        ...,
        description="The search query for tweets.",
    )


class TwitterAdvancedSearchTool(BaseTool):
    """Tool that gets tweets from Twitter."""

    name = "Twitter Advanced Search"

    slug = "get_tweets"

    description = "Get tweets for a given query."

    args_schema: Type[TwitterAdvancedSearchToolSchema] = TwitterAdvancedSearchToolSchema

    tool_id = "e831f03f-3282-4d5c-ae01-d7274515194d"

    def _run():
        pass

    async def _arun(
        self,
        query: str,  # run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Tweet message and return."""
        start_time = time.time()
        date_filter = (
            self.tool_manager.date_filter
            if self.tool_manager
            else get_specified_date_filter(DateFilterType.PAST_WEEK)
        )
        start_date = date_filter.start_date.date()
        end_date = date_filter.end_date.date()

        client = TwitterScraperActor()

        tweets = await client.get_tweets_advanced(
            urls=[],
            start=start_date,
            end=end_date,
            searchTerms=["Crypto"],
        )

        result = {
            "data": tweets,
            "result_count": len(tweets),
        }

        bt.logging.info(
            "================================== Twitter Advanced Search Results ==================================="
        )
        bt.logging.info(tweets)
        bt.logging.info(
            "================================== Twitter Advanced Search Results ===================================="
        )

        if self.tool_manager:
            self.tool_manager.twitter_data = result
            self.tool_manager.twitter_prompt_analysis = {}

        execution_time = (time.time() - start_time) / 60
        bt.logging.info("==========================================")
        bt.logging.info(
            f"Twitter Advanced Search Execution time is: {execution_time} minutes"
        )
        bt.logging.info("==========================================")

        return (result, {})

    async def send_event(self, send: Send, response_streamer, data):
        if not data:
            return

        tweets, prompt_analysis = data

        # Send prompt_analysis
        if prompt_analysis:
            prompt_analysis_response_body = {
                "type": "prompt_analysis",
                "content": prompt_analysis.dict(),
            }

            await send(
                {
                    "type": "http.response.body",
                    "body": json.dumps(prompt_analysis_response_body).encode("utf-8"),
                    "more_body": True,
                }
            )
            bt.logging.info("Prompt Analysis sent")

        if tweets:
            tweets_amount = tweets.get("result_count", 0)

            tweets_response_body = {"type": "tweets", "content": tweets}
            response_streamer.more_body = False

            await send(
                {
                    "type": "http.response.body",
                    "body": json.dumps(tweets_response_body).encode("utf-8"),
                    "more_body": False,
                }
            )
            bt.logging.info(f"Tweet data sent. Number of tweets: {tweets_amount}")
