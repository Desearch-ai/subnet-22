from typing import Type
import bittensor as bt
from pydantic import BaseModel, Field
from desearch.tools.base import BaseTool
from desearch.tools.search.scrapingdog_google_search import ScrapingDogGoogleSearch


class ArxivSearchSchema(BaseModel):
    query: str = Field(
        ...,
        description="The search query for ArXiv search.",
    )


class ArxivSearchTool(BaseTool):
    """Tool that searches the Arxiv API."""

    name = "ArXiv Search"

    slug = "arxiv-search"

    description = (
        "A wrapper around Arxiv.org "
        "Useful for when you need to answer questions about Physics, Mathematics, "
        "Computer Science, Quantitative Biology, Quantitative Finance, Statistics, "
        "Electrical Engineering, and Economics "
        "from scientific articles on arxiv.org. "
        "Input should be a search query."
    )

    args_schema: Type[ArxivSearchSchema] = ArxivSearchSchema

    tool_id = "58e41492-40e2-40f4-b548-c72a3b36ac72"

    async def _arun(self, query: str) -> str:
        """Search Arxiv and return the results."""
        search = ScrapingDogGoogleSearch(
            site="arxiv.org",
            query_suffix="-inurl:pdf",
        )
        return await search.search(query)

    async def send_event(self, send, response_streamer, data):
        if not data:
            return

        await response_streamer.send_event("arxiv_search", data)

        bt.logging.info("ArXiv search results data sent")
