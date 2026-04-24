"""
Concurrency ramp/decay logic for verified miner capacity.

All miners start at verified=1. Ramp up on quality, cut on failure.
Freeze miners that oscillate (declare high, can't deliver).
"""

from datetime import datetime, timedelta, timezone

import bittensor as bt

from desearch.miner_config import SEARCH_TYPES
from neurons.validators.scoring import miner_db

# Ease up on threshold before resolving issues in relevance models, as it's impossible to ramp up now.
RAMP_RATE = 0.05
DECAY_FACTOR = 0.8
QUALITY_THRESHOLD = 0.2
HARD_CAP = 100
FREEZE_FAILURES = 4
FREEZE_HOURS = 4

UNREACHABLE_FAILURE_THRESHOLD = 1
UNREACHABLE_DECAY_FACTOR = 0.9
UNREACHABLE_DECAY_INTERVAL_SEC = 5 * 60


async def get_verified(uid: int, search_type: str) -> int:
    return await miner_db.get_verified(uid, search_type)


async def get_all_verified(search_type: str) -> dict[int, int]:
    return await miner_db.get_all_verified(search_type)


async def register_miner(
    uid: int,
    search_type: str,
    declared: int,
    hotkey: str,
    coldkey: str,
) -> None:
    await miner_db.register_miner(
        uid=uid,
        search_type=search_type,
        declared=declared,
        hotkey=hotkey,
        coldkey=coldkey,
    )


async def update_after_scoring(
    uid: int,
    search_type: str,
    quality: float,
    window_start: str,
) -> None:
    row = await miner_db.get_concurrency_row(uid, search_type)
    if row is None:
        bt.logging.warning(
            f"[Capacity] update_after_scoring skipped — no row for "
            f"uid={uid} {search_type} (miner never registered?)"
        )
        return

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

    await miner_db.insert_window(
        uid=uid,
        search_type=search_type,
        window_start=window_start,
        hotkey=row["hotkey"],
        coldkey=row["coldkey"],
        quality_score=quality,
        passed=passed,
        verified_concurrency=new_verified,
    )

    new_frozen_until = frozen_until
    if not passed:
        fail_count = await miner_db.count_failed_windows(uid, search_type, FREEZE_HOURS)
        if fail_count >= FREEZE_FAILURES and not is_frozen:
            new_frozen_until = (now + timedelta(hours=FREEZE_HOURS)).isoformat()
            bt.logging.warning(
                f"[Capacity] Freezing uid={uid} {search_type} for "
                f"{FREEZE_HOURS}h ({fail_count} failures in {FREEZE_HOURS}h)"
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


async def note_call_result(uid: int, search_type: str, success: bool) -> None:
    """Record the outcome of a single dendrite call to a miner axon. After
    ``UNREACHABLE_FAILURE_THRESHOLD`` consecutive failures the miner is
    flagged unreachable and pulled from organic routing; the next success
    clears the flag."""

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
                bt.logging.warning(
                    f"[Capacity] uid={uid} {search_type} marked unreachable "
                    f"after {UNREACHABLE_FAILURE_THRESHOLD} consecutive failures"
                )
    except Exception as e:
        bt.logging.error(
            f"[Capacity] note_call_result failed uid={uid} {search_type}: {e}"
        )


async def decay_unreachable_tick() -> None:
    """Apply 10% earned-concurrency decay per ``UNREACHABLE_DECAY_INTERVAL_SEC``
    elapsed since the last tick for every miner currently marked unreachable.
    Catches up multiple intervals in one call so that longer outages compound
    without requiring the loop to fire on every interval."""

    now = datetime.now(timezone.utc)

    for search_type in SEARCH_TYPES:
        rows = await miner_db.get_unreachable_rows(search_type)
        for row in rows:
            last_tick_iso = row["last_decay_at"] or row["unreachable_since"]
            if not last_tick_iso:
                continue
            last_tick = datetime.fromisoformat(last_tick_iso)
            elapsed = (now - last_tick).total_seconds()
            ticks = int(elapsed // UNREACHABLE_DECAY_INTERVAL_SEC)
            if ticks <= 0:
                continue
            new_verified = row["verified"]
            for _ in range(ticks):
                new_verified = max(1, int(new_verified * UNREACHABLE_DECAY_FACTOR))
            new_last_decay = (
                last_tick + timedelta(seconds=ticks * UNREACHABLE_DECAY_INTERVAL_SEC)
            ).isoformat()
            await miner_db.apply_decay_tick(
                row["uid"], search_type, new_verified, new_last_decay
            )
            if new_verified != row["verified"]:
                bt.logging.info(
                    f"[Capacity] unreachable uid={row['uid']} {search_type}: "
                    f"verified {row['verified']}->{new_verified} ({ticks} ticks)"
                )
