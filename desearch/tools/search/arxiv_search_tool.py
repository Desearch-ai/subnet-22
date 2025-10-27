from typing import Optional, Type

import json
import bittensor as bt
from pydantic import BaseModel, Field
import arxiv
from desearch.tools.base import BaseTool


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
        client = arxiv.Client()

        search = arxiv.Search(
            query=query, max_results=10, sort_by=arxiv.SortCriterion.Relevance
        )

        results = []

        for r in client.results(search):
            results.append(
                {
                    "title": r.title,
                    "arxiv_url": r.entry_id,
                }
            )

        return results

    async def send_event(self, send, response_streamer, data):
        if not data:
            return

        search_results_response_body = {
            "type": "arxiv_search",
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

        bt.logging.info("ArXiv search results data sent")
