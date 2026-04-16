"""
SQLite persistence for miner concurrency state and scoring window history.

Module-level functions operating on a shared async connection.
Initialize once at startup with ``await initialize(db_path)``.
"""

from datetime import datetime, timedelta, timezone
from typing import Optional

import aiosqlite
import bittensor as bt

_db: Optional[aiosqlite.Connection] = None

STALENESS_HOURS = 24
RETENTION_DAYS = 3

_SCHEMA = """
CREATE TABLE IF NOT EXISTS miner_concurrency (
    uid              INTEGER NOT NULL,
    search_type      TEXT    NOT NULL,
    verified         INTEGER NOT NULL DEFAULT 1,
    declared         INTEGER NOT NULL DEFAULT 1,
    quality_avg      REAL    NOT NULL DEFAULT 0.0,
    frozen_until     TEXT,
    updated_at       TEXT    NOT NULL,
    PRIMARY KEY (uid, search_type)
);

CREATE TABLE IF NOT EXISTS scoring_windows (
    uid              INTEGER NOT NULL,
    search_type      TEXT    NOT NULL,
    window_start     TEXT    NOT NULL,
    quality_score    REAL    NOT NULL,
    passed           INTEGER NOT NULL DEFAULT 1,
    created_at       TEXT    NOT NULL,
    PRIMARY KEY (uid, search_type, window_start)
);
"""


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


async def initialize(db_path: str) -> None:
    global _db
    _db = await aiosqlite.connect(db_path)
    _db.row_factory = aiosqlite.Row
    await _db.execute("PRAGMA journal_mode=WAL")

    for statement in _SCHEMA.strip().split(";"):
        statement = statement.strip()
        if statement:
            await _db.execute(statement)
    await _db.commit()

    await _purge_old_history()
    await _decay_stale_verified()

    bt.logging.info(f"[MinerDB] Initialized at {db_path}")


async def close() -> None:
    global _db
    if _db:
        await _db.close()
        _db = None


async def _purge_old_history() -> None:
    cutoff = (datetime.now(timezone.utc) - timedelta(days=RETENTION_DAYS)).isoformat()
    await _db.execute("DELETE FROM scoring_windows WHERE created_at < ?", (cutoff,))
    await _db.commit()


async def _decay_stale_verified() -> None:
    """Reset verified→1 for miners not updated within STALENESS_HOURS."""
    cutoff = (datetime.now(timezone.utc) - timedelta(hours=STALENESS_HOURS)).isoformat()
    result = await _db.execute(
        "UPDATE miner_concurrency SET verified = 1 WHERE updated_at < ?", (cutoff,)
    )
    if result.rowcount:
        bt.logging.info(
            f"[MinerDB] Decayed {result.rowcount} stale miners to verified=1"
        )
    await _db.commit()


async def get_verified(uid: int, search_type: str) -> int:
    cursor = await _db.execute(
        "SELECT verified FROM miner_concurrency WHERE uid = ? AND search_type = ?",
        (uid, search_type),
    )
    row = await cursor.fetchone()
    return row["verified"] if row else 1


async def get_all_verified(search_type: str) -> dict[int, int]:
    cursor = await _db.execute(
        "SELECT uid, verified FROM miner_concurrency WHERE search_type = ?",
        (search_type,),
    )
    return {row["uid"]: row["verified"] async for row in cursor}


async def get_concurrency_row(uid: int, search_type: str) -> Optional[dict]:
    cursor = await _db.execute(
        "SELECT * FROM miner_concurrency WHERE uid = ? AND search_type = ?",
        (uid, search_type),
    )
    row = await cursor.fetchone()
    return dict(row) if row else None


async def upsert_concurrency(
    uid: int,
    search_type: str,
    verified: int,
    declared: int,
    quality_avg: float,
    frozen_until: Optional[str] = None,
) -> None:
    await _db.execute(
        """
        INSERT INTO miner_concurrency (uid, search_type, verified, declared, quality_avg, frozen_until, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(uid, search_type) DO UPDATE SET
            verified = excluded.verified,
            declared = excluded.declared,
            quality_avg = excluded.quality_avg,
            frozen_until = excluded.frozen_until,
            updated_at = excluded.updated_at
        """,
        (uid, search_type, verified, declared, quality_avg, frozen_until, _now_iso()),
    )
    await _db.commit()


async def register_miner(uid: int, search_type: str, declared: int) -> None:
    """Insert-or-ignore: only creates row if miner is new."""
    await _db.execute(
        """
        INSERT OR IGNORE INTO miner_concurrency (uid, search_type, verified, declared, quality_avg, updated_at)
        VALUES (?, ?, 1, ?, 0.0, ?)
        """,
        (uid, search_type, declared, _now_iso()),
    )
    # Always update declared (miner may have changed config)
    await _db.execute(
        "UPDATE miner_concurrency SET declared = ?, updated_at = ? WHERE uid = ? AND search_type = ?",
        (declared, _now_iso(), uid, search_type),
    )
    await _db.commit()


async def insert_window(
    uid: int,
    search_type: str,
    window_start: str,
    quality_score: float,
    passed: bool,
) -> None:
    await _db.execute(
        """
        INSERT OR REPLACE INTO scoring_windows (uid, search_type, window_start, quality_score, passed, created_at)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (uid, search_type, window_start, quality_score, int(passed), _now_iso()),
    )
    await _db.commit()


async def count_failed_windows(
    uid: int, search_type: str, since_hours: int = 12
) -> int:
    cutoff = (datetime.now(timezone.utc) - timedelta(hours=since_hours)).isoformat()
    cursor = await _db.execute(
        """
        SELECT COUNT(*) as cnt FROM scoring_windows
        WHERE uid = ? AND search_type = ? AND passed = 0 AND created_at >= ?
        """,
        (uid, search_type, cutoff),
    )
    row = await cursor.fetchone()
    return row["cnt"] if row else 0
