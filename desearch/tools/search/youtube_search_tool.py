from typing import Optional, Type
from pydantic import BaseModel, Field
from youtube_search import YoutubeSearch
import json
import bittensor as bt
from desearch.tools.base import BaseTool


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
        result = YoutubeSearch(search_terms=query, max_results=10)

        videos = [
            {"url": f"https://www.youtube.com{video['url_suffix']}", **video}
            for video in result.videos
        ]

        return videos

    async def send_event(self, send, response_streamer, data):
        if not data:
            return

        search_results_response_body = {
            "type": "youtube_search",
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

        bt.logging.info("Youtube search results data sent")
