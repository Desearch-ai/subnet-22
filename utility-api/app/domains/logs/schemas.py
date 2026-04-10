from datetime import datetime
from typing import Any
from uuid import UUID

from app.domains.dataset.enums import SearchType
from app.domains.logs.enums import QueryKind
from pydantic import BaseModel, Field


class MinerResponseLogCreate(BaseModel):
    query_kind: QueryKind = Field(
        description="Log origin. `organic` is customer-originated input; `scoring` is validator scoring traffic."
    )
    search_type: SearchType = Field(
        description="Search input type such as ai_search, x_search, x_post_by_id, x_posts_by_urls, or web_search."
    )
    netuid: int

    scoring_epoch_start: datetime | None = None

    miner_uid: int | None = None
    miner_hotkey: str
    miner_coldkey: str | None = None

    validator_uid: int | None = None
    validator_hotkey: str
    validator_coldkey: str | None = None

    request_query: str
    status_code: int | None = None
    process_time: float | None = None
    total_reward: float | None = None

    response_payload: dict[str, Any] = Field(default_factory=dict)
    reward_payload: dict[str, Any] | None = None


class SaveMinerResponseLogsRequest(BaseModel):
    logs: list[MinerResponseLogCreate]


class SaveMinerResponseLogsResponse(BaseModel):
    inserted: int


class ScoringValidatorLogResponse(BaseModel):
    id: UUID
    created_at: datetime
    validator_uid: int | None = None
    validator_hotkey: str
    validator_coldkey: str | None = None
    status_code: int | None = None
    process_time: float | None = None
    total_reward: float | None = None
    response_payload: dict[str, Any] = Field(default_factory=dict)
    reward_payload: dict[str, Any] | None = None


class ScoringLogGroupResponse(BaseModel):
    scoring_epoch_start: datetime
    miner_uid: int | None = None
    miner_hotkey: str
    miner_coldkey: str | None = None
    search_type: SearchType
    request_query: str
    validator_count: int
    reward_min: float | None = None
    reward_max: float | None = None
    reward_avg: float | None = None
    logs: list[ScoringValidatorLogResponse]


class GetScoringLogsResponse(BaseModel):
    groups: list[ScoringLogGroupResponse]


class OrganicLogResponse(BaseModel):
    id: UUID
    created_at: datetime
    search_type: SearchType
    miner_uid: int | None = None
    miner_hotkey: str
    miner_coldkey: str | None = None
    validator_uid: int | None = None
    validator_hotkey: str
    validator_coldkey: str | None = None
    request_query: str
    status_code: int | None = None
    process_time: float | None = None


class BatchOrganicSearchRequest(BaseModel):
    search_type: SearchType
    created_at_start: datetime
    created_at_end: datetime
    queries: list[str] = Field(
        description="List of exact request_query strings to match against organic logs."
    )
    validator_hotkey: str | None = None
    miner_coldkey: str | None = None
    miner_hotkey: str | None = None


class BatchOrganicMatchResult(BaseModel):
    request_query: str
    logs: list[OrganicLogResponse]


class BatchOrganicSearchResponse(BaseModel):
    matches: list[BatchOrganicMatchResult]
    total_matched_queries: int
    total_logs: int
