"""Per-UID, per-lane verified-concurrency ramping."""

from datetime import datetime, timedelta, timezone
from typing import Optional, Protocol

import bittensor as bt

from desearch.miner_config import LANES, lane_from_key, lane_key
from neurons.validators.scoring.constants import DEFAULT_PER_UID, QUALITY_THRESHOLDS
from neurons.validators.scoring import miner_db

QUALITY_EMA_ALPHA = 0.5

HARD_CAP_PER_UID = 100
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


def passes_lane_gate(lane_quality: float, declared: int, search_type) -> bool:
    return declared > 0 and lane_quality >= QUALITY_THRESHOLDS[search_type]


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
        passed=quality_avg >= QUALITY_THRESHOLDS[lane_from_key(search_type)[0]],
        verified_concurrency=allocated,
    )

    await miner_db.upsert_quality_avg(
        uid=uid,
        search_type=search_type,
        quality_avg=quality_avg,
    )


async def ramp_after_epoch(uids: list[int]) -> None:
    """Ramp each lane on its own quality so a weak mode cannot drag the others."""
    if not uids:
        return

    state = await miner_db.get_quality_state_bulk(uids)
    if not state:
        return

    known = {lane_key(lane) for lane in LANES}
    updates_by_lane: dict[str, dict[int, int]] = {key: {} for key in known}
    passed = 0
    total = 0

    for uid, by_lane in state.items():
        for key, row in by_lane.items():
            if key not in known:
                continue
            search_type, _mode = lane_from_key(key)
            total += 1
            lane_passed = passes_lane_gate(
                row["quality_avg"], row["declared"], search_type
            )
            passed += int(lane_passed)
            new_verified = next_verified(row["verified"], row["declared"], lane_passed)
            if new_verified != row["verified"]:
                updates_by_lane[key][uid] = new_verified

    for key, updates in updates_by_lane.items():
        if updates:
            await miner_db.bulk_update_verified(key, updates)

    bt.logging.info(f"[Capacity] ramp_after_epoch: {passed}/{total} lanes passed gate")


def lanes_for(search_type, mode=None) -> list[str]:
    if mode is not None:
        return [lane_key((search_type, mode))]
    return [lane_key(lane) for lane in LANES if lane[0] == search_type]


async def note_call_result(uid: int, search_type, success: bool, mode=None) -> None:
    """A failed query marks only its own lane; a dead axon marks every lane."""
    try:
        for key in lanes_for(search_type, mode):
            if success:
                recovered = await miner_db.record_call_success(uid, key)
                if recovered:
                    bt.logging.info(
                        f"[Capacity] uid={uid} {key} recovered from unreachable"
                    )
            else:
                newly = await miner_db.record_call_failure(
                    uid, key, UNREACHABLE_FAILURE_THRESHOLD
                )
                if newly:
                    if _router is not None:
                        _router.mark_unreachable(uid, search_type)
                    bt.logging.warning(
                        f"[Capacity] uid={uid} {key} marked unreachable after "
                        f"{UNREACHABLE_FAILURE_THRESHOLD} consecutive failures"
                    )
    except Exception as e:
        bt.logging.error(
            f"[Capacity] note_call_result failed uid={uid} {search_type}: {e}"
        )


async def decay_unreachable_tick() -> None:
    """Apply 10% verified decay per elapsed 5-min interval for unreachable miners."""
    now = datetime.now(timezone.utc)

    for lane in LANES:
        search_type = lane_key(lane)
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
