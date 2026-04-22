from typing import Type

import bittensor as bt
from pydantic import BaseModel, Field

from desearch.tools.base import BaseTool
from desearch.tools.search.scrapingdog_google_search import ScrapingDogGoogleSearch


class HackerNewsSearchSchema(BaseModel):
    query: str = Field(
        ...,
        description="The search query for Hacker News search.",
    )


class HackerNewsSearchTool(BaseTool):
    """Tool for the HackerNews API."""

    name = "Hacker News Search"
    slug = "hacker-news-search"
    description = (
        "A wrapper around Hacker News. Useful for searching Hacker News for posts."
    )
    args_schema: Type[HackerNewsSearchSchema] = HackerNewsSearchSchema
    tool_id = "b6cf5471-2f58-4a86-b0de-b5b3653c086f"

    async def _arun(self, query: str) -> str:
        """Search Hacker News and return the results."""
        search = ScrapingDogGoogleSearch(site="news.ycombinator.com")
        return await search.search(query)

    async def send_event(self, send, response_streamer, data):
        if not data:
            return

        await response_streamer.send_event("hacker_news_search", data)

        bt.logging.info("Hacker News search results data sent")
