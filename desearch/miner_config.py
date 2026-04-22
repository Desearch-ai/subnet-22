import json
import os

import bittensor as bt
from pydantic import BaseModel, Field

MAX_CONCURRENCY_PER_TYPE = 100
SEARCH_TYPES = ("web_search", "x_search", "ai_search")

class ConcurrencyConfig(BaseModel):
    """Per-search-type, per-validator concurrency ceiling."""

    web_search: int = Field(default=1, ge=1, le=MAX_CONCURRENCY_PER_TYPE)
    x_search: int = Field(default=1, ge=1, le=MAX_CONCURRENCY_PER_TYPE)
    ai_search: int = Field(default=1, ge=1, le=MAX_CONCURRENCY_PER_TYPE)


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
