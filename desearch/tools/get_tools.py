from typing import List
from desearch.tools.base import BaseToolkit, BaseTool
from desearch.tools.twitter.twitter_toolkit import TwitterToolkit
from desearch.tools.search.search_toolkit import SearchToolkit
from desearch.tools.reddit.reddit_toolkit import RedditToolkit
from desearch.tools.hacker_news.hacker_news_toolkit import HackerNewsToolkit

TOOLKITS: List[BaseToolkit] = [
    SearchToolkit(),
    TwitterToolkit(),
    RedditToolkit(),
    HackerNewsToolkit(),
]


def get_all_tools():
    """Return a list of all tools."""
    result: List[BaseTool] = []

    for toolkit in TOOLKITS:
        tools = toolkit.get_tools()
        result.extend(tools)

    return result


def find_toolkit_by_tool_name(tool_name: str):
    """Return the toolkit that contains the tool with the given name."""
    for toolkit in TOOLKITS:
        for tool in toolkit.get_tools():
            if tool.name == tool_name:
                return toolkit

    return None


def find_toolkit_by_name(toolkit_name: str):
    """Return the toolkit with the given name."""
    for toolkit in TOOLKITS:
        if toolkit.name == toolkit_name:
            return toolkit

    return None
