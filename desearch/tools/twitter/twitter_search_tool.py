import json
from typing import Type
import bittensor as bt
from pydantic import BaseModel, Field
from starlette.types import Send
from desearch.tools.base import BaseTool
from desearch.services.twitter_prompt_analyzer import TwitterPromptAnalyzer
from desearch.dataset.date_filters import get_specified_date_filter, DateFilterType
from desearch.tools.twitter.twitter_utils import generalize_tweet_structure


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
        openai_query_model = (
            self.tool_manager.miner.config.miner.openai_query_model
            if self.tool_manager
            else "gpt-3.5-turbo-0125"
        )
        openai_fix_query_model = (
            self.tool_manager.miner.config.miner.openai_fix_query_model
            if self.tool_manager
            else "gpt-4-1106-preview"
        )
        date_filter = (
            self.tool_manager.date_filter
            if self.tool_manager
            else get_specified_date_filter(DateFilterType.PAST_WEEK)
        )

        client = TwitterPromptAnalyzer(
            openai_query_model=openai_query_model,
            openai_fix_query_model=openai_fix_query_model,
        )

        result, prompt_analysis = await client.analyse_prompt_and_fetch_tweets(
            query,
            date_filter=date_filter,
        )

        bt.logging.info(
            "================================== Prompt analysis ==================================="
        )
        bt.logging.info(prompt_analysis)
        bt.logging.info(
            "================================== Prompt analysis ===================================="
        )

        if self.tool_manager:
            self.tool_manager.twitter_data = result
            self.tool_manager.twitter_prompt_analysis = prompt_analysis

        return (result, prompt_analysis)

    async def send_event(self, send: Send, response_streamer, data):
        if not data:
            return

        tweets, prompt_analysis = data

        # # Send prompt_analysis
        # if prompt_analysis:
        #     prompt_analysis_response_body = {
        #         "type": "prompt_analysis",
        #         "content": prompt_analysis.dict(),
        #     }

        #     await send(
        #         {
        #             "type": "http.response.body",
        #             "body": json.dumps(prompt_analysis_response_body).encode("utf-8"),
        #             "more_body": True,
        #         }
        #     )
        #     bt.logging.info("Prompt Analysis sent")

        if tweets:
            modified_tweets = generalize_tweet_structure(tweets=tweets)
            tweets_response_body = {"type": "tweets", "content": modified_tweets}
            response_streamer.more_body = False

            await send(
                {
                    "type": "http.response.body",
                    "body": json.dumps(tweets_response_body).encode("utf-8"),
                    "more_body": False,
                }
            )
            bt.logging.info(
                f"Tweet data sent. Number of tweets: {len(modified_tweets)}"
            )
