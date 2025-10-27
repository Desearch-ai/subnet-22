from typing import Optional, Type
import json

import bittensor as bt
from pydantic import BaseModel, Field

from desearch.tools.search.wikipedia_api_wrapper import WikipediaAPIWrapper
from desearch.tools.base import BaseTool


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
        wikipedia = WikipediaAPIWrapper()
        return wikipedia.run(query)

    async def send_event(self, send, response_streamer, data):
        if not data:
            return

        search_results_response_body = {
            "type": "wikipedia_search",
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

        bt.logging.info("Wikipedia search results data sent")


if __name__ == "__main__":
    tool = WikipediaSearchTool()
    result = tool._arun("george washington")
    print(result)
