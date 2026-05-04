import enum


class QueryKind(str, enum.Enum):
    ORGANIC = "organic"
    SCORING = "scoring"


class SearchType(str, enum.Enum):
    """Type of search a log entry belongs to."""

    AI_SEARCH = "ai_search"
    X_SEARCH = "x_search"
    X_POST_BY_ID = "x_post_by_id"
    X_POSTS_BY_URLS = "x_posts_by_urls"
    WEB_SEARCH = "web_search"
