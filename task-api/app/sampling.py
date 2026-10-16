"""Which uploads validators check: a share of each miner's, drawn from a block hash nobody knew at upload."""

from __future__ import annotations

import hashlib
import time

SHARE = 0.05
# Drawn checks an hour across all miners, so validators keep up whatever the volume.
CHECKS_PER_HOUR = 400
ALL = "all"
NEW_HOTKEY_PASSES = 10
RECHECK_UPLOADS = 10
NEW, RECHECK, DRAW = "new", "recheck", "draw"
HOUR = 3600


def draw(seed: str, task_id: str, etag: str) -> float:
    digest = hashlib.sha256(f"{seed}:{task_id}:{etag}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


def pick_reason(
    drawn: float,
    uploads_last_hour: float,
    passes: int,
    recheck_left: int,
    share: float = SHARE,
) -> str | None:
    """Every upload of a new hotkey and of one re-checked after a fail; otherwise the share, and about one an hour at least."""
    if passes < NEW_HOTKEY_PASSES:
        return NEW
    if recheck_left > 0:
        return RECHECK
    if drawn < max(share, 1 / max(uploads_last_hour, 1.0)):
        return DRAW
    return None


def hour_of(at: float | None = None) -> int:
    return int((at or time.time()) // HOUR)


async def note_upload(redis, hotkey: str, at: float | None = None) -> None:
    for counted in (hotkey, ALL):
        key = f"uploads:{counted}:{hour_of(at)}"
        if await redis.incr(key) == 1:
            await redis.expire(key, 3 * HOUR)


def budget_share(share: float, per_hour: float, all_last_hour: float) -> float:
    """The drawn share, lowered so all miners' draws together stay within the hourly budget."""
    if per_hour <= 0:
        return share
    return min(share, per_hour / max(all_last_hour, 1.0))


async def uploads_last_hour(redis, hotkey: str, now: float | None = None) -> float:
    """This hour's uploads plus the unexpired part of the last hour's."""
    now = now or time.time()
    hour = hour_of(now)
    this_hour = float(await redis.get(f"uploads:{hotkey}:{hour}") or 0)
    last_hour = float(await redis.get(f"uploads:{hotkey}:{hour - 1}") or 0)
    return this_hour + last_hour * (1 - (now % HOUR) / HOUR)
