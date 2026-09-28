import asyncio
import logging

import bittensor as bt
import numpy as np

EMISSION_CONTROL_HOTKEY = "5CUu1QhvrfyMDBELUPJLt4c7uJFbi7TKqDHkS1Zz41oD4dyP"
# Each task family's part of the emission; the rest goes to the burn hotkey.
# Embedding pays nothing until Desearch's own model ships and embed tasks open.
POOLS = {"crawl": 0.5, "embed": 0.0}
SET_WEIGHTS_ATTEMPTS = 9
SET_WEIGHTS_RETRY_S = 45
VERSION_KEY = 2**64 - 9

log = logging.getLogger("validator")


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


def process_weights(
    weights: np.ndarray, min_allowed: int, max_limit: float
) -> tuple[list[int], list[float]]:
    """The nonzero weights, capped at the subnet's limit and normalised to sum to one."""
    uids = [int(uid) for uid in np.flatnonzero(weights > 0)]
    if len(uids) < min_allowed:
        raise ValueError(
            f"{len(uids)} weights, the subnet needs at least {min_allowed}"
        )
    values = np.array([float(weights[uid]) for uid in uids], dtype=np.float64)
    values /= values.sum()
    if max_limit < 1.0:
        values = np.minimum(values, max_limit)
        values /= values.sum()
    return uids, values.tolist()


async def set_weights(neuron, weights: np.ndarray) -> bool:
    netuid = neuron.config.netuid
    chain = neuron.subtensor
    try:
        uids, processed = process_weights(
            weights,
            await chain.hyperparameters.min_allowed_weights(netuid=netuid),
            await chain.hyperparameters.max_weight_limit(netuid=netuid),
        )
    except ValueError as why:
        log.error(f"Not setting weights: {why}")
        return False
    log.info(
        "Setting weights: "
        + " | ".join(
            f"{uid}={weight:.4f}" for uid, weight in zip(uids, processed, strict=True)
        )
    )
    intent = bt.SetWeights(
        netuid=netuid, uids=uids, weights=processed, version_key=VERSION_KEY
    )

    for attempt in range(1, SET_WEIGHTS_ATTEMPTS + 1):
        try:
            result = await chain.execute(intent, neuron.wallet)
            success, message = result.success, describe(result)
        except Exception as exc:
            success, message = False, f"{type(exc).__name__}: {exc}"
        if success:
            log.info(f"Set weights on attempt {attempt}: {message}")
            return True
        log.warning(f"Setting weights failed on attempt {attempt}: {message}")
        await asyncio.sleep(SET_WEIGHTS_RETRY_S)

    log.error(f"Could not set weights after {SET_WEIGHTS_ATTEMPTS} attempts")
    return False


def describe(result) -> str:
    error = getattr(result, "error", None)
    if error is not None:
        return f"{getattr(error, 'code', error)}: {getattr(error, 'remediation', '')}"
    return getattr(result, "message", "") or "ok"
