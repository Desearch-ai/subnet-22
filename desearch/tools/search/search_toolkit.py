from abc import ABC
from typing import List
from desearch.tools.base import BaseToolkit, BaseTool
from .web_search_tool import WebSearchTool
from .wikipedia_search_tool import WikipediaSearchTool
from .youtube_search_tool import YoutubeSearchTool
from .arxiv_search_tool import ArxivSearchTool


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
