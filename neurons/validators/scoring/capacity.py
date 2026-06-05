"""Per-UID verified-concurrency ramping gated on serving all 3 search types."""

from datetime import datetime, timedelta, timezone
from typing import Optional, Protocol

import bittensor as bt

from desearch.miner_config import SEARCH_TYPES
from neurons.validators.scoring import miner_db

QUALITY_EMA_ALPHA = 0.5

DEFAULT_PER_UID = 1
HARD_CAP_PER_UID = 100
QUALITY_THRESHOLDS: dict[str, float] = {
    "ai_search": 0.45,
    "x_search": 0.60,
    "web_search": 0.60,
}
RAMP_FRACTION = 0.10
DECAY_FRACTION = 0.10

UNREACHABLE_FAILURE_THRESHOLD = 1
UNREACHABLE_DECAY_FACTOR = 0.9
UNREACHABLE_DECAY_INTERVAL_SEC = 5 * 60


class _RouterKillSwitch(Protocol):
    def mark_unreachable(self, uid: int, search_type: str) -> None: ...


_router: Optional[_RouterKillSwitch] = None


def set_router(router: _RouterKillSwitch) -> None:
    """Register the routing weight cache so we can zero a UID's weight the
    moment it flips to unreachable, instead of waiting up to 10 minutes for
    the next metagraph sweep to refresh the cache."""
    global _router
    _router = router


def next_verified(current: int, declared: int, all_pass: bool) -> int:
    declared = max(declared, DEFAULT_PER_UID)
    if all_pass:
        step = max(1, int(declared * RAMP_FRACTION))
        return min(current + step, declared, HARD_CAP_PER_UID)
    step = max(1, int(declared * DECAY_FRACTION))
    return max(DEFAULT_PER_UID, current - step)


def passes_combined_gate(
    ema_by_type: dict[str, float],
    declared_by_type: dict[str, int],
) -> bool:
    for t, thr in QUALITY_THRESHOLDS.items():
        if declared_by_type.get(t, 0) <= 0:
            return False
        if ema_by_type.get(t, 0.0) < thr:
            return False
    return True


async def record_window_quality(
    uid: int,
    search_type: str,
    quality: float,
    window_start: str,
    allocated: int,
) -> None:
    """Update EMA and log the window. Ramping deferred to ``ramp_after_epoch``."""
    row = await miner_db.get_concurrency_row(uid, search_type)
    if row is None:
        bt.logging.warning(
            f"[Capacity] record_window_quality skipped — no row for "
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
        passed=quality_avg >= QUALITY_THRESHOLDS[search_type],
        verified_concurrency=allocated,
    )

    await miner_db.upsert_quality_avg(
        uid=uid,
        search_type=search_type,
        quality_avg=quality_avg,
    )


async def ramp_after_epoch(uids: list[int]) -> None:
    """Evaluate the combined gate per UID and persist verified across all 3 types."""
    if not uids:
        return

    state = await miner_db.get_quality_state_bulk(uids)
    if not state:
        return

    updates_by_type: dict[str, dict[int, int]] = {t: {} for t in QUALITY_THRESHOLDS}
    pass_count = 0

    for uid, by_type in state.items():
        ema = {t: row["quality_avg"] for t, row in by_type.items()}
        declared = {t: row["declared"] for t, row in by_type.items()}
        all_pass = passes_combined_gate(ema, declared)
        if all_pass:
            pass_count += 1

        for t, row in by_type.items():
            new_v = next_verified(row["verified"], row["declared"], all_pass)
            if new_v != row["verified"]:
                updates_by_type[t][uid] = new_v

    for t, updates in updates_by_type.items():
        if updates:
            await miner_db.bulk_update_verified(t, updates)

    bt.logging.info(
        f"[Capacity] ramp_after_epoch: gate passed for {pass_count}/{len(state)} UIDs"
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


async def decay_unreachable_tick() -> None:
    """Apply 10% verified decay per elapsed 5-min interval for unreachable miners."""
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
