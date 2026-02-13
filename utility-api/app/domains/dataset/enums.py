import enum


class SearchType(str, enum.Enum):
    """Type of search a question is suitable for."""

    AI_SEARCH = "ai_search"
    X_SEARCH = "x_search"
    WEB_SEARCH = "web_search"


class AISearchTool(str, enum.Enum):
    """Tool used by AI search miners to gather sources."""

    TWITTER = "twitter"
    WEB = "web"
    REDDIT = "reddit"
    HACKER_NEWS = "hacker_news"
    YOUTUBE = "youtube"
    ARXIV = "arxiv"
    WIKIPEDIA = "wikipedia"
