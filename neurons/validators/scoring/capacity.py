"""
Concurrency ramp/decay logic for verified miner capacity.

All miners start at verified=1. Ramp up on quality, cut on failure.
Freeze miners that oscillate (declare high, can't deliver).
"""

from datetime import datetime, timedelta, timezone

import bittensor as bt

from neurons.validators.scoring import miner_db

RAMP_RATE = 0.05
DECAY_FACTOR = 0.7
QUALITY_THRESHOLD = 0.5
HARD_CAP = 100
FREEZE_FAILURES = 4
FREEZE_HOURS = 12


async def get_verified(uid: int, search_type: str) -> int:
    return await miner_db.get_verified(uid, search_type)


async def get_all_verified(search_type: str) -> dict[int, int]:
    return await miner_db.get_all_verified(search_type)


async def register_miner(uid: int, search_type: str, declared: int) -> None:
    await miner_db.register_miner(uid, search_type, declared)


async def update_after_scoring(
    uid: int,
    search_type: str,
    quality: float,
    window_start: str,
) -> None:
    row = await miner_db.get_concurrency_row(uid, search_type)
    if row is None:
        await miner_db.register_miner(uid, search_type, declared=1)
        row = await miner_db.get_concurrency_row(uid, search_type)

    verified = row["verified"]
    declared = row["declared"]
    frozen_until = row["frozen_until"]

    now = datetime.now(timezone.utc)
    is_frozen = frozen_until and datetime.fromisoformat(frozen_until) > now
    passed = quality >= QUALITY_THRESHOLD

    if passed and not is_frozen:
        increment = max(1, int(declared * RAMP_RATE))
        new_verified = min(verified + increment, declared, HARD_CAP)
    elif not passed:
        new_verified = max(1, int(verified * DECAY_FACTOR))
    else:
        new_verified = verified

    await miner_db.insert_window(uid, search_type, window_start, quality, passed)

    new_frozen_until = frozen_until
    if not passed:
        fail_count = await miner_db.count_failed_windows(uid, search_type, FREEZE_HOURS)
        if fail_count >= FREEZE_FAILURES and not is_frozen:
            new_frozen_until = (now + timedelta(hours=FREEZE_HOURS)).isoformat()
            bt.logging.warning(
                f"[Capacity] Freezing uid={uid} {search_type} for {FREEZE_HOURS}h "
                f"({fail_count} failures in {FREEZE_HOURS}h)"
            )

    quality_avg = 0.8 * row["quality_avg"] + 0.2 * quality

    await miner_db.upsert_concurrency(
        uid=uid,
        search_type=search_type,
        verified=new_verified,
        declared=declared,
        quality_avg=quality_avg,
        frozen_until=new_frozen_until,
    )

    if new_verified != verified:
        bt.logging.info(
            f"[Capacity] uid={uid} {search_type}: "
            f"verified {verified}->{new_verified} (quality={quality:.3f})"
        )
