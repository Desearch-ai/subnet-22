"""
Quality-share allocation for synthetic queries.

Each search type has a fixed per-epoch budget split across miners whose
quality EMA crosses ``QUALITY_THRESHOLD``, weighted by ``quality^k``. UIDs
below threshold get only the floor — they have to demonstrate quality
across multiple windows before participating in the share split.
``declared`` is an optional miner-set ceiling.
"""

from typing import Optional, Protocol

import bittensor as bt

from neurons.validators.scoring import miner_db

QUALITY_EMA_ALPHA = 0.3
QUALITY_SHARE_EXPONENT = 2.0

SYNTHETIC_BUDGET_PER_TYPE = 250
DEFAULT_PER_UID = 1
HARD_CAP_PER_UID = 100
QUALITY_THRESHOLD = 0.30
RAMP_STEP_PCT = 0.10

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


def allocate_synthetic_budget(
    quality_by_uid: dict[int, float],
    declared_by_uid: dict[int, int],
    prev_alloc_by_uid: dict[int, int],
) -> dict[int, int]:
    """Split the budget across above-threshold UIDs by quality^k, capped per
    UID by ``prev + RAMP_STEP_PCT * declared`` so volume ramps incrementally.
    Below-threshold UIDs get only the floor."""
    if not quality_by_uid:
        return {}

    eligible = {uid: q for uid, q in quality_by_uid.items() if q >= QUALITY_THRESHOLD}

    weights = {
        uid: max(0.0, q) ** QUALITY_SHARE_EXPONENT for uid, q in eligible.items()
    }
    total = sum(weights.values())

    out: dict[int, int] = {uid: DEFAULT_PER_UID for uid in quality_by_uid}

    if total <= 0:
        return out

    for uid, w in weights.items():
        share = round(SYNTHETIC_BUDGET_PER_TYPE * w / total)
        declared = max(declared_by_uid.get(uid, DEFAULT_PER_UID), DEFAULT_PER_UID)
        prev = max(prev_alloc_by_uid.get(uid, DEFAULT_PER_UID), DEFAULT_PER_UID)
        ramp_step = max(1, int(declared * RAMP_STEP_PCT))
        ramp_cap = prev + ramp_step
        ceiling = min(HARD_CAP_PER_UID, declared, ramp_cap)
        out[uid] = min(DEFAULT_PER_UID + int(share), ceiling)
    return out


async def update_after_scoring(
    uid: int,
    search_type: str,
    quality: float,
    window_start: str,
    allocated: int,
) -> None:
    """Update a miner's quality EMA from one scoring window. ``allocated`` is
    the per-UID synthetic budget the scheduler used for the window being
    scored — passed in because the DB column has already been overwritten
    with the next epoch's allocation by the time scoring fires."""
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
