"""
Per-UID verified-concurrency ramping.

Each miner's ``verified`` grows ``RAMP_FRACTION`` of declared per scoring
window when quality EMA stays at or above ``QUALITY_THRESHOLD``, and decays
the same step when it falls below. ``verified`` is the synthetic allocation
for the next epoch — bounded by ``declared`` and ``HARD_CAP_PER_UID``.
"""

from typing import Optional, Protocol

import bittensor as bt

from neurons.validators.scoring import miner_db

QUALITY_EMA_ALPHA = 0.3

DEFAULT_PER_UID = 1
HARD_CAP_PER_UID = 100
QUALITY_THRESHOLD = 0.30
RAMP_FRACTION = 0.10
DECAY_FRACTION = 0.20

UNREACHABLE_FAILURE_THRESHOLD = 1


class _RouterKillSwitch(Protocol):
    def mark_unreachable(self, uid: int, search_type: str) -> None: ...


_router: Optional[_RouterKillSwitch] = None


def set_router(router: _RouterKillSwitch) -> None:
    """Register the routing weight cache so we can zero a UID's weight the
    moment it flips to unreachable, instead of waiting up to 10 minutes for
    the next metagraph sweep to refresh the cache."""
    global _router
    _router = router


def next_verified(current: int, declared: int, quality_avg: float) -> int:
    """Ramp by ``RAMP_FRACTION``, decay by ``DECAY_FRACTION`` (faster exit than entry)."""
    declared = max(declared, DEFAULT_PER_UID)
    if quality_avg >= QUALITY_THRESHOLD:
        step = max(1, int(declared * RAMP_FRACTION))
        return min(current + step, declared, HARD_CAP_PER_UID)
    step = max(1, int(declared * DECAY_FRACTION))
    return max(DEFAULT_PER_UID, current - step)


async def update_after_scoring(
    uid: int,
    search_type: str,
    quality: float,
    window_start: str,
    allocated: int,
) -> None:
    """Update quality EMA and ramp ``verified`` for one scoring window."""
    row = await miner_db.get_concurrency_row(uid, search_type)
    if row is None:
        bt.logging.warning(
            f"[Capacity] update_after_scoring skipped — no row for "
            f"uid={uid} {search_type} (miner never registered?)"
        )
        return

    quality_avg = (1 - QUALITY_EMA_ALPHA) * row[
        "quality_avg"
    ] + QUALITY_EMA_ALPHA * quality

    new_verified = next_verified(row["verified"], row["declared"], quality_avg)

    await miner_db.insert_window(
        uid=uid,
        search_type=search_type,
        window_start=window_start,
        hotkey=row["hotkey"],
        coldkey=row["coldkey"],
        quality_score=quality,
        passed=quality >= QUALITY_THRESHOLD,
        verified_concurrency=allocated,
    )

    await miner_db.upsert_quality_avg(
        uid=uid,
        search_type=search_type,
        quality_avg=quality_avg,
    )
    await miner_db.bulk_update_verified(search_type, {uid: new_verified})


async def note_call_result(uid: int, search_type: str, success: bool) -> None:
    """Record the outcome of a single dendrite call. After
    ``UNREACHABLE_FAILURE_THRESHOLD`` consecutive failures the miner is flagged
    unreachable and pulled from organic routing; the next success clears it."""

    try:
        if success:
            recovered = await miner_db.record_call_success(uid, search_type)
            if recovered:
                bt.logging.info(
                    f"[Capacity] uid={uid} {search_type} recovered from unreachable"
                )
        else:
            newly = await miner_db.record_call_failure(
                uid, search_type, UNREACHABLE_FAILURE_THRESHOLD
            )
            if newly:
                if _router is not None:
                    _router.mark_unreachable(uid, search_type)
                bt.logging.warning(
                    f"[Capacity] uid={uid} {search_type} marked unreachable "
                    f"after {UNREACHABLE_FAILURE_THRESHOLD} consecutive failures"
                )
    except Exception as e:
        bt.logging.error(
            f"[Capacity] note_call_result failed uid={uid} {search_type}: {e}"
        )
