from datetime import datetime
from typing import Any
from uuid import UUID

from app.domains.dataset.enums import SearchType
from app.domains.logs.enums import QueryKind
from pydantic import BaseModel, Field


class MinerResponseLogCreate(BaseModel):
    query_kind: QueryKind
    search_type: SearchType
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
