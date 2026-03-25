from typing import Type

from pydantic import BaseModel, Field
import bittensor as bt

from desearch.tools.base import BaseTool
from desearch.tools.search.scrapingdog_google_search import ScrapingDogGoogleSearch


class RedditSearchSchema(BaseModel):
    query: str = Field(
        ...,
        description="The search query for Reddit search.",
    )


class RedditSearchTool(BaseTool):
    """Tool for the Reddit API."""

    name = "Reddit Search"

    slug = "reddit-search"

    description = "A wrapper around Reddit." "Useful for searching Reddit for posts."

    args_schema: Type[RedditSearchSchema] = RedditSearchSchema

    tool_id = "043489f8-ef05-4151-8849-7f954e4910be"

    def _run():
        pass

    async def _arun(self, query: str) -> str:
        """Search Reddit and return the results."""
        search = ScrapingDogGoogleSearch(
            site="reddit.com",
            query_suffix="inurl:comments",
        )
        return await search.search(query)

    async def send_event(self, send, response_streamer, data):
        if not data:
            return

        await response_streamer.send_event("reddit_search", data)

        bt.logging.info("Reddit search results data sent")
