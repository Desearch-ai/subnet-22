from enum import Enum


class RewardModelType(Enum):
    summary_relavance_match = "summary_relavance_match"
    twitter_content_relevance = "twitter_content_relevance"
    twitter_basic_search_content_relevance = "twitter_basic_search_content_relevance"
    search_content_relevance = "search_content_relevance"
    content_relevance = "content_relevance"
    performance_score = "performance_score"


class RewardScoringType(Enum):
    summary_relevance_score_template = "summary_relevance_score_template"
    link_content_relevance_template = "link_content_relevance_template"
    search_relevance_score_template = "search_relevance_score_template"
    performance_score_template = "performance_score_template"
