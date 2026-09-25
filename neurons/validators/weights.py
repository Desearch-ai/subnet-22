import asyncio

import bittensor as bt
import numpy as np
from bittensor.utils.weight_utils import process_weights

EMISSION_CONTROL_HOTKEY = "5CUu1QhvrfyMDBELUPJLt4c7uJFbi7TKqDHkS1Zz41oD4dyP"
# Each task family's part of the emission; the rest goes to the burn hotkey.
POOLS = {"crawl": 0.25, "embed": 0.25}
SET_WEIGHTS_ATTEMPTS = 9
SET_WEIGHTS_RETRY_S = 45
VERSION_KEY = 2**64 - 9


def weights_from_shares(
    hotkeys: list[str], pools: dict[str, dict[str, float]]
) -> np.ndarray:
    weights = np.zeros(len(hotkeys), dtype=np.float32)
    uid_of = {hotkey: uid for uid, hotkey in enumerate(hotkeys)}
    unpaid = 1.0
    for pool, part in POOLS.items():
        paid = {
            uid_of[hotkey]: share
            for hotkey, share in pools.get(pool, {}).items()
            if hotkey in uid_of and hotkey != EMISSION_CONTROL_HOTKEY and share > 0
        }
        total = sum(paid.values())
        if not total:
            continue
        for uid, share in paid.items():
            weights[uid] += part * share / total
        unpaid -= part
    burn = uid_of.get(EMISSION_CONTROL_HOTKEY)
    if burn is not None:
        weights[burn] += unpaid
    return weights


async def set_weights(neuron, weights: np.ndarray) -> bool:
    netuid = neuron.config.netuid
    uids, processed = process_weights(
        uids=neuron.metagraph.uids,
        weights=weights,
        num_neurons=int(neuron.metagraph.n),
        min_allowed_weights=await neuron.subtensor.min_allowed_weights(netuid=netuid),
        max_weight_limit=await neuron.subtensor.max_weight_limit(netuid=netuid),
    )
    bt.logging.info(
        "Setting weights: "
        + " | ".join(
            f"{uid}={weight:.4f}" for uid, weight in zip(uids, processed, strict=True)
        )
    )

    for attempt in range(1, SET_WEIGHTS_ATTEMPTS + 1):
        try:
            success, message = await neuron.subtensor.set_weights(
                wallet=neuron.wallet,
                netuid=netuid,
                uids=uids,
                weights=processed,
                wait_for_inclusion=False,
                wait_for_finalization=False,
                version_key=VERSION_KEY,
            )
        except Exception as exc:
            success, message = False, f"{type(exc).__name__}: {exc}"
        if success:
            bt.logging.success(f"Set weights on attempt {attempt}: {message}")
            return True
        bt.logging.warning(f"Setting weights failed on attempt {attempt}: {message}")
        await asyncio.sleep(SET_WEIGHTS_RETRY_S)

    bt.logging.error(f"Could not set weights after {SET_WEIGHTS_ATTEMPTS} attempts")
    return False
