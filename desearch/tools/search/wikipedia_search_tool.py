from typing import Type

import bittensor as bt
from pydantic import BaseModel, Field

from desearch.tools.base import BaseTool
from desearch.tools.search.scrapingdog_google_search import ScrapingDogGoogleSearch


class WikipediaSearchSchema(BaseModel):
    query: str = Field(
        ...,
        description="The search query for Wikipedia search.",
    )


class WikipediaSearchTool(BaseTool):
    """Tool for the Wikipedia API."""

    name = "Wikipedia Search"

    slug = "wikipedia-search"

    description = (
        "A wrapper around Wikipedia. "
        "Useful for when you need to answer general questions about "
        "people, places, companies, facts, historical events, or other subjects. "
        "Input should be a search query."
    )

    args_schema: Type[WikipediaSearchSchema] = WikipediaSearchSchema

    tool_id = "eb161647-b858-4863-801f-ba7d2e380601"

    async def _arun(self, query: str) -> str:
        """Search Wikipedia and return the results."""
        search = ScrapingDogGoogleSearch(site="wikipedia.org")
        return await search.search(query)

    async def send_event(self, send, response_streamer, data):
        if not data:
            return

        await response_streamer.send_event("wikipedia_search", data)

        bt.logging.info("Wikipedia search results data sent")


if __name__ == "__main__":
    tool = WikipediaSearchTool()
    result = tool._arun("george washington")
    print(result)
