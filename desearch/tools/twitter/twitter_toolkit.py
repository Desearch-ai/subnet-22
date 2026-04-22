from abc import ABC
from typing import List
from desearch.tools.base import BaseToolkit, BaseTool
from desearch.tools.twitter.twitter_search_tool import TwitterSearchTool


class TwitterToolkit(BaseToolkit, ABC):
    name: str = "Twitter Toolkit"
    description: str = "Toolkit containing tools for retrieving tweets."
    slug: str = "twitter"
    toolkit_id: str = "0e0ae6fb-0f1c-4d00-bc84-1feb2a6824c6"

    def get_tools(self) -> List[BaseTool]:
        return [TwitterSearchTool()]
