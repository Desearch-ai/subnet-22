import random
from typing import List, Optional

import bittensor as bt
from bittensor.core.metagraph import AsyncMetagraph

from desearch.miner_config import SEARCH_TYPES
from neurons.validators.scoring import miner_db
from neurons.validators.scoring.weights import EMISSION_CONTROL_HOTKEY

QUALITY_FLOOR = 0.1


class UIDManager:
    """
    Routes organic requests to miners weighted by quality * verified concurrency
    per search type. Snapshots are refreshed on metagraph resync.
    """

    metagraph: AsyncMetagraph

    def __init__(self) -> None:
        self.available_uids: List[int] = []
        self.weights_by_type: dict[str, dict[int, float]] = {
            st: {} for st in SEARCH_TYPES
        }

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

        for search_type in SEARCH_TYPES:
            rows = await miner_db.get_all_concurrency_data(search_type)
            unreachable = await miner_db.get_unreachable_uids(search_type)
            weights: dict[int, float] = {}
            for uid in available_uids:
                if uid in unreachable:
                    weights[uid] = 0.0
                    continue
                quality_avg, verified = rows.get(uid, (0.0, 1))
                weights[uid] = max(quality_avg, QUALITY_FLOOR) * max(verified, 1)
            self.weights_by_type[search_type] = weights

        bt.logging.info(
            f"[UIDManager] Resynced weights for {len(available_uids)} miners "
            f"across {len(SEARCH_TYPES)} search types"
        )

    def get_miner_uid(self, search_type: Optional[str] = None) -> int:
        if not self.available_uids:
            raise RuntimeError("UIDManager has no available UIDs")

        if search_type and search_type in self.weights_by_type:
            weights_map = self.weights_by_type[search_type]
            uids = self.available_uids
            weights = [weights_map.get(uid, 0.0) for uid in uids]
            if sum(weights) > 0:
                return random.choices(uids, weights=weights, k=1)[0]

        return random.choice(self.available_uids)
