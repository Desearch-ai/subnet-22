import random
from typing import List, Optional

import bittensor as bt
from bittensor.core.metagraph import AsyncMetagraph

from desearch.miner_config import LANES, Lane, SearchType, lane_key
from desearch.protocol import SearchMode
from neurons.validators.scoring import miner_db
from neurons.validators.scoring.constants import (
    QUALITY_EXPONENT,
    VOLUME_EXPONENT,
)
from neurons.validators.scoring.weights import EMISSION_CONTROL_HOTKEY

QUALITY_FLOOR = 0.1
RAMP_EVIDENCE_VERIFIED = 2
MIN_MIGRATION_POOL = 2


class UIDManager:
    """
    Routes organic requests to miners weighted by quality * verified concurrency
    per lane. Snapshots are refreshed on metagraph resync.
    """

    metagraph: AsyncMetagraph

    def __init__(self) -> None:
        self.available_uids: List[int] = []
        self.weights_by_lane: dict[Lane, dict[int, float]] = {}

    def _top_half_by_incentive(self, available_uids: List[int]) -> set[int]:
        available_set = set(available_uids)
        target_size = max(MIN_MIGRATION_POOL, len(available_uids) // 2)
        ranked = (-self.metagraph.I).argsort().tolist()
        pool: set[int] = set()
        for uid in ranked:
            uid = int(uid) if not isinstance(uid, int) else uid
            if uid in available_set:
                pool.add(uid)
                if len(pool) >= target_size:
                    break
        return pool

    @staticmethod
    def lane_for(search_type: SearchType, mode: Optional[SearchMode]) -> Lane:
        if search_type != SearchType.AI_SEARCH:
            return (search_type, None)
        return (search_type, SearchMode(mode) if mode else SearchMode.BALANCED)

    async def resync(
        self,
        available_uids: List[int],
        metagraph: Optional[AsyncMetagraph] = None,
    ) -> None:
        if metagraph is not None:
            self.metagraph = metagraph

        if not available_uids:
            self.available_uids = []
            return

        if EMISSION_CONTROL_HOTKEY:
            emission_control_uid = next(
                (
                    neuron.uid
                    for neuron in self.metagraph.neurons
                    if neuron.hotkey == EMISSION_CONTROL_HOTKEY
                ),
                None,
            )
            available_uids = [
                uid for uid in available_uids if uid != emission_control_uid
            ]

        self.available_uids = available_uids
        migration_pool = self._top_half_by_incentive(available_uids)

        lane_modes: dict[str, str] = {}
        for lane in LANES:
            key = lane_key(lane)
            rows = await miner_db.get_all_concurrency_data(key)
            unreachable = await miner_db.get_unreachable_uids(key)

            any_ramped = any(v >= RAMP_EVIDENCE_VERIFIED for _q, v in rows.values())
            lane_modes[key] = "ramped" if any_ramped else "migration"

            weights: dict[int, float] = {}
            for uid in available_uids:
                if uid in unreachable:
                    weights[uid] = 0.0
                    continue
                if any_ramped:
                    quality_avg, verified = rows.get(uid, (0.0, 1))
                    weights[uid] = (
                        max(quality_avg, QUALITY_FLOOR) ** QUALITY_EXPONENT
                        * max(verified, 1) ** VOLUME_EXPONENT
                    )
                else:
                    weights[uid] = 1.0 if uid in migration_pool else 0.0
            self.weights_by_lane[lane] = weights

        bt.logging.info(
            f"[UIDManager] Resynced {len(available_uids)} reachable "
            f"(migration pool={len(migration_pool)}): {lane_modes}"
        )

    def mark_unreachable(self, uid: int, search_type: SearchType) -> None:
        for lane, weights in self.weights_by_lane.items():
            if lane[0] == search_type and weights.get(uid, 0.0) > 0:
                weights[uid] = 0.0

    async def drop_unreachable(self) -> None:
        for lane in LANES:
            for uid in await miner_db.get_unreachable_uids(lane_key(lane)):
                self.mark_unreachable(uid, lane[0])

    def _pick(self, weights_map: dict[int, float]) -> Optional[int]:
        uids = self.available_uids
        weights = [weights_map.get(uid, 0.0) for uid in uids]
        if sum(weights) > 0:
            return random.choices(uids, weights=weights, k=1)[0]
        return None

    def get_miner_uid(
        self,
        search_type: Optional[SearchType] = None,
        mode: Optional[SearchMode] = None,
    ) -> int:
        if not self.available_uids:
            raise RuntimeError("UIDManager has no available UIDs")

        if search_type:
            weights = self.weights_by_lane.get(self.lane_for(search_type, mode))
            if weights:
                selected = self._pick(weights)
                if selected is not None:
                    return selected

        return random.choice(self.available_uids)
