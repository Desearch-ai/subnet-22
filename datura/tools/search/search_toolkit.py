from abc import ABC
from typing import List
from datura.tools.base import BaseToolkit, BaseTool
from .web_search_tool import WebSearchTool
from .wikipedia_search_tool import WikipediaSearchTool
from .youtube_search_tool import YoutubeSearchTool
from .arxiv_search_tool import ArxivSearchTool
from .search_summary import summarize_search_data, prepare_search_data_for_summary


TOOLS = [
    WebSearchTool(),
    WikipediaSearchTool(),
    YoutubeSearchTool(),
    ArxivSearchTool(),
]


class SearchToolkit(BaseToolkit, ABC):
    name: str = "Search Toolkit"
    description: str = (
        "Toolkit containing tools for performing web, youtube, wikipedia and other searches."
    )

    slug: str = "web-search"
    toolkit_id: str = "fed46dde-ee8e-420b-a1bb-4a161aa01dca"

    def get_tools(self) -> List[BaseTool]:
        return TOOLS

    async def summarize(self, prompt, model, data):
        return await summarize_search_data(
            prompt=prompt,
            model=model,
            data=prepare_search_data_for_summary(data),
        )
