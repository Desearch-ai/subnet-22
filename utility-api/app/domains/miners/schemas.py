from typing import Optional

from pydantic import BaseModel


class ValidatorInfo(BaseModel):
    id: str
    uid: int
    hotkey: str
    label: str
    online: bool


class MinerTypeState(BaseModel):
    verified: int
    declared: int
    quality_avg: float
    frozen_until: Optional[str] = None
    unreachable_since: Optional[str] = None


class ScoringWindow(BaseModel):
    window_start: str
    quality_score: float
    passed: bool
    verified_concurrency: int


class MinerListItem(BaseModel):
    hotkey: str
    uid: int
    coldkey: str
    by_validator: dict[str, dict[str, MinerTypeState]]


class MinerListResponse(BaseModel):
    validators: list[ValidatorInfo]
    miners: list[MinerListItem]


class MinerDetail(BaseModel):
    hotkey: str
    uid: int
    coldkey: str
    per_type: dict[str, MinerTypeState]
    windows: dict[str, list[ScoringWindow]]


class ValidatorMinerView(BaseModel):
    validator: ValidatorInfo
    detail: Optional[MinerDetail] = None


class MinerDetailResponse(BaseModel):
    views: list[ValidatorMinerView]
