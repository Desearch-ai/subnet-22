from types import SimpleNamespace
from unittest.mock import AsyncMock

import numpy as np
import pytest

from desearch.miner_config import LANES
from neurons.validators.proxy import uid_manager
from neurons.validators.proxy.uid_manager import UIDManager
from neurons.validators.scoring.constants import (
    QUALITY_EXPONENT,
    VOLUME_EXPONENT,
)


def _metagraph():
    return SimpleNamespace(
        I=np.array([0.0, 0.3, 0.2, 0.1]),
        neurons=[SimpleNamespace(uid=uid, hotkey=f"h{uid}") for uid in range(4)],
    )


async def test_ramped_organic_routing_scales_with_verified_capacity(
    monkeypatch,
):
    rows = {1: (0.7, 100), 2: (0.7, 50), 3: (0.7, 50)}
    monkeypatch.setattr(
        uid_manager.miner_db,
        "get_all_concurrency_data",
        AsyncMock(return_value=rows),
    )
    monkeypatch.setattr(
        uid_manager.miner_db,
        "get_unreachable_uids",
        AsyncMock(return_value=set()),
    )

    manager = UIDManager()
    await manager.resync([1, 2, 3], _metagraph())

    for lane in LANES:
        weights = manager.weights_by_lane[lane]
        assert weights[1] >= weights[2] + weights[3]
        assert weights[1] / (weights[2] + weights[3]) == pytest.approx(
            2 ** (VOLUME_EXPONENT - 1)
        )


async def test_ramped_organic_routing_uses_quality_exponent(monkeypatch):
    rows = {1: (0.8, 10), 2: (0.7, 10)}
    monkeypatch.setattr(
        uid_manager.miner_db,
        "get_all_concurrency_data",
        AsyncMock(return_value=rows),
    )
    monkeypatch.setattr(
        uid_manager.miner_db,
        "get_unreachable_uids",
        AsyncMock(return_value=set()),
    )

    manager = UIDManager()
    await manager.resync([1, 2], _metagraph())

    for lane in LANES:
        weights = manager.weights_by_lane[lane]
        assert weights[1] / weights[2] == pytest.approx((0.8 / 0.7) ** QUALITY_EXPONENT)
