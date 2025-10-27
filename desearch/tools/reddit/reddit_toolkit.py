from abc import ABC
from typing import List
from desearch.tools.base import BaseToolkit, BaseTool
from .reddit_summary import summarize_reddit_data, prepare_reddit_data_for_summary
from .reddit_search_tool import RedditSearchTool


class RedditToolkit(BaseToolkit, ABC):
    name: str = "Reddit Toolkit"
    description: str = "Toolkit containing tools for retrieving tweets."
    slug: str = "reddit"
    toolkit_id: str = "c6efe1a4-6cdf-404f-b6e4-92ed6c524f0a"

    def get_tools(self) -> List[BaseTool]:
        return [RedditSearchTool()]

    async def summarize(self, prompt, model, data, system_message):
        data = next(iter(data.values()))
        return await summarize_reddit_data(
            prompt=prompt,
            model=model,
            filtered_posts=data,
            user_system_message=system_message,
        )
