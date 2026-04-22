import bittensor as bt
from typing import Type
from pydantic import BaseModel, Field
from desearch.tools.base import BaseTool
from .scrapingdog_google_search import ScrapingDogGoogleSearch


class WebSearchSchema(BaseModel):
    query: str = Field(
        ...,
        description="The search query for web search.",
    )


class WebSearchTool(BaseTool):
    name = "Web Search"

    slug = "web_search"

    description = (
        "This tool performs web search and extracts relevant snippets and webpages. "
        "It's particularly useful for staying updated with current events and finding quick answers to your queries."
    )

    args_schema: Type[WebSearchSchema] = WebSearchSchema

    tool_id = "a66b3b20-d0a2-4b53-a775-197bc492e816"

    def _run():
        pass

    async def _arun(
        self,
        query: str,
    ):
        """Search web and return the results."""
        search = ScrapingDogGoogleSearch()

        try:
            return await search.search(query)
        except Exception as err:
            bt.logging.error(f"Could not perform web search: {err}")
            return []

    async def send_event(self, send, response_streamer, data):
        if not data:
            return

        await response_streamer.send_event("search", data)

        bt.logging.info("Web search results data sent")
