import json
import os
from enum import Enum
from typing import Dict, Optional, Union

import bittensor as bt
from pydantic import BaseModel, Field, field_validator

from desearch.protocol import SearchMode

MAX_CONCURRENCY_PER_TYPE = 100


class SearchType(str, Enum):
    X_SEARCH = "x_search"
    AI_SEARCH = "ai_search"


SEARCH_TYPES = (SearchType.X_SEARCH, SearchType.AI_SEARCH)

AI_MODES = (SearchMode.FAST, SearchMode.BALANCED, SearchMode.DEEP)

Lane = tuple[SearchType, Optional[SearchMode]]

LANES: tuple[Lane, ...] = tuple(
    [(SearchType.AI_SEARCH, mode) for mode in AI_MODES] + [(SearchType.X_SEARCH, None)]
)


def lane_key(lane: Lane) -> str:
    search_type, mode = lane
    key = SearchType(search_type).value
    return f"{key}:{SearchMode(mode).value}" if mode else key


def lane_from_key(key: str) -> Lane:
    search_type, _, mode = key.partition(":")
    return (SearchType(search_type), SearchMode(mode) if mode else None)


class ConcurrencyConfig(BaseModel):
    """Per-lane, per-validator concurrency ceiling."""

    x_search: int = Field(default=1, ge=1, le=MAX_CONCURRENCY_PER_TYPE)
    ai_search: Union[int, Dict[SearchMode, int]] = Field(default=1)

    @field_validator("ai_search")
    @classmethod
    def _check_ai_search(cls, value):
        values = value.values() if isinstance(value, dict) else [value]
        for item in values:
            if not 1 <= item <= MAX_CONCURRENCY_PER_TYPE:
                raise ValueError(
                    f"ai_search concurrency must be 1..{MAX_CONCURRENCY_PER_TYPE}"
                )
        return value

    def ai_by_mode(self) -> Dict[SearchMode, int]:
        if isinstance(self.ai_search, dict):
            return {mode: self.ai_search.get(mode, 1) for mode in AI_MODES}
        return {mode: self.ai_search for mode in AI_MODES}

    def by_lane(self) -> Dict[Lane, int]:
        declared = {
            (SearchType.AI_SEARCH, mode): value
            for mode, value in self.ai_by_mode().items()
        }
        declared[(SearchType.X_SEARCH, None)] = self.x_search
        return declared


class MinerManifest(BaseModel):
    concurrency: ConcurrencyConfig = Field(default_factory=ConcurrencyConfig)


def normalize_miner_manifest(data: dict) -> MinerManifest:
    return MinerManifest.model_validate(data)


def default_miner_manifest() -> MinerManifest:
    return MinerManifest()


def load_miner_manifest(path: str) -> MinerManifest:
    expanded_path = os.path.expanduser(path)

    if not os.path.exists(expanded_path):
        bt.logging.warning(
            f"Miner config file not found at {expanded_path}. Using default manifest."
        )
        return default_miner_manifest()

    with open(expanded_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    manifest = normalize_miner_manifest(data)
    bt.logging.info(
        f"Loaded miner manifest from {expanded_path}: "
        f"concurrency={manifest.concurrency.model_dump()}"
    )
    return manifest
