import asyncio
import logging
import math

import bittensor as bt
import numpy as np

from desearch.kinds import CRAWL, EMBED

EMISSION_CONTROL_HOTKEY = "5CUu1QhvrfyMDBELUPJLt4c7uJFbi7TKqDHkS1Zz41oD4dyP"
EMISSION_CONTROL_PERC = 0.8
CRAWL_PERC = 1.0
EMBED_PERC = 0.0
assert math.isclose(CRAWL_PERC + EMBED_PERC, 1.0)
SET_WEIGHTS_ATTEMPTS = 9
SET_WEIGHTS_RETRY_S = 45
VERSION_KEY = 2**64 - 9
U16_MAX = 65535

log = logging.getLogger("validator")


def weights_from_shares(
    hotkeys: list[str], shares: dict[str, dict[str, float]]
) -> np.ndarray:
    """Each pool's part split by its miners' shares; a pool nobody earned is burned too."""
    weights = np.zeros(len(hotkeys), dtype=np.float32)
    uid_of = {hotkey: uid for uid, hotkey in enumerate(hotkeys)}
    for_miners = 1.0 - EMISSION_CONTROL_PERC
    paid = pay_pool(weights, uid_of, shares.get(CRAWL, {}), for_miners * CRAWL_PERC)
    paid += pay_pool(weights, uid_of, shares.get(EMBED, {}), for_miners * EMBED_PERC)
    burn = uid_of.get(EMISSION_CONTROL_HOTKEY)
    if burn is not None:
        weights[burn] += 1.0 - paid
    return weights


def pay_pool(
    weights: np.ndarray, uid_of: dict[str, int], shares: dict[str, float], perc: float
) -> float:
    """Adds one pool's part of the emission to its miners' weights and returns what it paid."""
    earned = {
        uid_of[hotkey]: share
        for hotkey, share in shares.items()
        if hotkey in uid_of and hotkey != EMISSION_CONTROL_HOTKEY and share > 0
    }
    total = sum(earned.values())
    if not total or not perc:
        return 0.0
    for uid, share in earned.items():
        weights[uid] += perc * share / total
    return perc


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
    limit = max_limit / U16_MAX if max_limit > 1.0 else max_limit
    if limit < 1.0:
        values = np.minimum(values, limit)
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
