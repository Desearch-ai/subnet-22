from typing import Type
from pydantic import BaseModel, Field
import bittensor as bt
from desearch.tools.base import BaseTool
from desearch.tools.search.scrapingdog_google_search import ScrapingDogGoogleSearch


class YoutubeSearchSchema(BaseModel):
    query: str = Field(
        ...,
        description="The search query for Youtube search.",
    )


class YoutubeSearchTool(BaseTool):
    """Tool for the Youtube API."""

    name = "Youtube Search"

    slug = "youtube-search"

    description = (
        "Useful for when you need to search videos on Youtube"
        "Input should be a search query."
    )

    args_schema: Type[YoutubeSearchSchema] = YoutubeSearchSchema

    tool_id = "8b7b6dad-e550-4a01-be51-aed785eda805"

    async def _arun(self, query: str) -> str:
        """Search Youtube and return the results."""
        search = ScrapingDogGoogleSearch(site="youtube.com/watch")
        return await search.search(query)

    async def send_event(self, send, response_streamer, data):
        if not data:
            return

        await response_streamer.send_event("youtube_search", data)

        bt.logging.info("Youtube search results data sent")
