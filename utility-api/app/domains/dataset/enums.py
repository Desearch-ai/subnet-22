import enum


class SearchType(str, enum.Enum):
    """Type of search a question is suitable for."""

    AI_SEARCH = "ai_search"
    X_SEARCH = "x_search"
    X_POST_BY_ID = "x_post_by_id"
    X_POSTS_BY_URLS = "x_posts_by_urls"
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
