from typing import Optional, Type
from pydantic import BaseModel, Field
from desearch.tools.search.serp_advanced_google_search import SerpAdvancedGoogleSearch
from desearch.tools.base import BaseTool
import json
import bittensor as bt


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
        search = SerpAdvancedGoogleSearch(
            site="news.ycombinator.com",
            language=self.tool_manager.language if self.tool_manager else "en",
            region=self.tool_manager.region if self.tool_manager else "us",
            date_filter=(
                self.tool_manager.google_date_filter if self.tool_manager else "qdr:w"
            ),
        )
        result = await search.run(query)
        result = search.process_response(result)
        return result

    async def send_event(self, send, response_streamer, data):
        if not data:
            return

        search_results_response_body = {
            "type": "hacker_news_search",
            "content": data,
        }

        response_streamer.more_body = False

        await send(
            {
                "type": "http.response.body",
                "body": json.dumps(search_results_response_body).encode("utf-8"),
                "more_body": False,
            }
        )

        bt.logging.info("Hacker News search results data sent")
