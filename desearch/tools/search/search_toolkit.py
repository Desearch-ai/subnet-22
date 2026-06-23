from abc import ABC
from typing import List
from desearch.tools.base import BaseToolkit, BaseTool
from .web_search_tool import WebSearchTool


TOOLS = [
    WebSearchTool(),
]


class SearchToolkit(BaseToolkit, ABC):
    name: str = "Search Toolkit"
    description: str = "Toolkit containing the web search tool."

    slug: str = "web-search"
    toolkit_id: str = "fed46dde-ee8e-420b-a1bb-4a161aa01dca"

    def get_tools(self) -> List[BaseTool]:
        return TOOLS
