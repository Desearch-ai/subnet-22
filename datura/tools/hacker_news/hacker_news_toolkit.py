from abc import ABC
from typing import List
from datura.tools.base import BaseToolkit, BaseTool
from .hacker_news_summary import (
    summarize_hacker_news_data,
    prepare_hacker_news_data_for_summary,
)
from .hacker_news_search_tool import HackerNewsSearchTool

TOOLS = [HackerNewsSearchTool()]


class HackerNewsToolkit(BaseToolkit, ABC):
    name: str = "Hacker News Toolkit"
    description: str = "Toolkit containing tools for searching hacker news."
    slug: str = "hacker-news"
    toolkit_id: str = "28a7dba6-c79b-4489-badc-d75948c37935"

    def get_tools(self) -> List[BaseTool]:
        return TOOLS

    async def summarize(self, prompt, model, data):
        data = next(iter(data.values()))
        return await summarize_hacker_news_data(
            prompt=prompt,
            model=model,
            filtered_posts=data,
        )
