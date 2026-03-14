from datetime import datetime
from typing import Any

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
