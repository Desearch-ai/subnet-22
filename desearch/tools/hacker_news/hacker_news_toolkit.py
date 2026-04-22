from abc import ABC
from typing import List

from desearch.tools.base import BaseTool, BaseToolkit

from .hacker_news_search_tool import HackerNewsSearchTool

TOOLS = [HackerNewsSearchTool()]


class HackerNewsToolkit(BaseToolkit, ABC):
    name: str = "Hacker News Toolkit"
    description: str = "Toolkit containing tools for searching hacker news."
    slug: str = "hacker-news"
    toolkit_id: str = "28a7dba6-c79b-4489-badc-d75948c37935"

    def get_tools(self) -> List[BaseTool]:
        return TOOLS
