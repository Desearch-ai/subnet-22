"""
SQLite persistence for miner concurrency state and scoring window history.

Module-level functions operate on a shared async connection.
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
    uid                   INTEGER NOT NULL,
    search_type           TEXT    NOT NULL,
    hotkey                TEXT    NOT NULL,
    coldkey               TEXT    NOT NULL,
    verified              INTEGER NOT NULL DEFAULT 1,
    declared              INTEGER NOT NULL DEFAULT 1,
    pending_declared      INTEGER,
    quality_avg           REAL    NOT NULL DEFAULT 0.0,
    frozen_until          TEXT,
    consecutive_failures  INTEGER NOT NULL DEFAULT 0,
    unreachable_since     TEXT,
    last_decay_at         TEXT,
    updated_at            TEXT    NOT NULL,
    PRIMARY KEY (uid, search_type)
);

CREATE INDEX IF NOT EXISTS idx_miner_concurrency_hotkey
    ON miner_concurrency (hotkey);

CREATE TABLE IF NOT EXISTS scoring_windows (
    uid              INTEGER NOT NULL,
    search_type      TEXT    NOT NULL,
    window_start     TEXT    NOT NULL,
    hotkey           TEXT    NOT NULL,
    coldkey          TEXT    NOT NULL,
    quality_score    REAL    NOT NULL,
    passed           INTEGER NOT NULL DEFAULT 1,
    verified_concurrency   INTEGER NOT NULL,
    created_at       TEXT    NOT NULL,
    PRIMARY KEY (uid, search_type, window_start)
);

CREATE INDEX IF NOT EXISTS idx_scoring_windows_created
    ON scoring_windows (created_at);

CREATE INDEX IF NOT EXISTS idx_scoring_windows_hotkey
    ON scoring_windows (hotkey, search_type, window_start);
"""


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


async def initialize(db_path: str, readonly: bool = False) -> None:
    """Open the shared async connection.

    ``readonly=True`` opens the DB via the ``file:…?mode=ro`` URI so a process
    that only serves reads (e.g. a public-API uvicorn worker) cannot mutate
    state owned by the neuron. The neuron process calls this with
    ``readonly=False`` and creates the schema + runs retention cleanup."""
    global _db
    if readonly:
        uri = f"file:{db_path}?mode=ro"
        _db = await aiosqlite.connect(uri, uri=True)
        _db.row_factory = aiosqlite.Row
        bt.logging.info(f"[MinerDB] Opened read-only at {db_path}")
        return

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


async def get_all_concurrency_data(
    search_type: str,
) -> dict[int, tuple[float, int]]:
    """Return {uid: (quality_avg, verified)} for all miners of a given search type."""
    cursor = await _db.execute(
        """
        SELECT uid, quality_avg, verified
        FROM miner_concurrency WHERE search_type = ?
        """,
        (search_type,),
    )
    return {row["uid"]: (row["quality_avg"], row["verified"]) async for row in cursor}


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
        UPDATE miner_concurrency
        SET verified = ?,
            declared = ?,
            quality_avg = ?,
            frozen_until = ?,
            updated_at = ?
        WHERE uid = ? AND search_type = ?
        """,
        (
            verified,
            declared,
            quality_avg,
            frozen_until,
            _now_iso(),
            uid,
            search_type,
        ),
    )
    await _db.commit()


async def register_miner(
    uid: int,
    search_type: str,
    declared: int,
    hotkey: str,
    coldkey: str,
) -> None:
    """Insert a new miner row, or refresh an existing one with the current
    hotkey/coldkey from the caller's metagraph snapshot.

    If the hotkey at this UID *changed* (deregister/re-register of a new
    miner under the same UID), the stale row is deleted first so the new
    holder starts fresh at verified=1 with no carried quality or flags.
    If the hotkey is unchanged, the UPSERT refreshes identity columns and
    stages a ``declared`` change into ``pending_declared`` (promoted at the
    next hour boundary by ``promote_pending_declared`` so mid-hour edits
    don't disturb an in-flight scoring window)."""
    now = _now_iso()
    await _db.execute(
        """
        DELETE FROM miner_concurrency
        WHERE uid = ? AND search_type = ? AND hotkey != ?
        """,
        (uid, search_type, hotkey),
    )
    await _db.execute(
        """
        INSERT INTO miner_concurrency
            (uid, search_type, hotkey, coldkey, verified, declared,
             quality_avg, updated_at)
        VALUES (?, ?, ?, ?, 1, ?, 0.0, ?)
        ON CONFLICT(uid, search_type) DO UPDATE SET
            coldkey = excluded.coldkey,
            updated_at = excluded.updated_at,
            pending_declared = CASE
                WHEN miner_concurrency.declared != excluded.declared
                    THEN excluded.declared
                ELSE NULL
            END
        """,
        (uid, search_type, hotkey, coldkey, declared, now),
    )
    await _db.commit()


async def promote_pending_declared() -> int:
    cursor = await _db.execute(
        """
        UPDATE miner_concurrency
        SET declared = pending_declared,
            pending_declared = NULL,
            updated_at = ?
        WHERE pending_declared IS NOT NULL
        """,
        (_now_iso(),),
    )
    await _db.commit()
    return cursor.rowcount or 0


async def insert_window(
    uid: int,
    search_type: str,
    window_start: str,
    hotkey: str,
    coldkey: str,
    quality_score: float,
    passed: bool,
    verified_concurrency: int,
) -> None:
    await _db.execute(
        """
        INSERT OR REPLACE INTO scoring_windows
            (uid, search_type, window_start, hotkey, coldkey,
             quality_score, passed, verified_concurrency, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            uid,
            search_type,
            window_start,
            hotkey,
            coldkey,
            quality_score,
            int(passed),
            verified_concurrency,
            _now_iso(),
        ),
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


async def record_call_success(uid: int, search_type: str) -> bool:
    """Clear consecutive_failures and unreachable_since for an already-registered
    miner. Returns ``True`` when this call ended an unreachable state so the
    caller can log the recovery. No-op (returns ``False``) if the row doesn't
    exist — registration is the exclusive job of ``register_miner``."""
    cursor = await _db.execute(
        """
        SELECT consecutive_failures, unreachable_since
        FROM miner_concurrency WHERE uid = ? AND search_type = ?
        """,
        (uid, search_type),
    )
    row = await cursor.fetchone()
    if row is None:
        return False
    if row["consecutive_failures"] == 0 and row["unreachable_since"] is None:
        return False

    was_unreachable = bool(row["unreachable_since"])
    await _db.execute(
        """
        UPDATE miner_concurrency
        SET consecutive_failures = 0,
            unreachable_since = NULL,
            last_decay_at = NULL,
            updated_at = ?
        WHERE uid = ? AND search_type = ?
        """,
        (_now_iso(), uid, search_type),
    )
    await _db.commit()
    return was_unreachable


async def record_call_failure(uid: int, search_type: str, threshold: int) -> bool:
    """Increment ``consecutive_failures`` and mark unreachable when the counter
    crosses ``threshold`` for the first time. Returns ``True`` on that
    transition. No-op (returns ``False``) if the row doesn't exist."""
    cursor = await _db.execute(
        """
        SELECT consecutive_failures, unreachable_since
        FROM miner_concurrency WHERE uid = ? AND search_type = ?
        """,
        (uid, search_type),
    )
    row = await cursor.fetchone()
    if row is None:
        return False

    now = _now_iso()
    new_count = row["consecutive_failures"] + 1
    was_unreachable = row["unreachable_since"] is not None
    flip = new_count >= threshold and not was_unreachable

    if flip:
        await _db.execute(
            """
            UPDATE miner_concurrency
            SET consecutive_failures = ?,
                unreachable_since = ?,
                last_decay_at = ?,
                updated_at = ?
            WHERE uid = ? AND search_type = ?
            """,
            (new_count, now, now, now, uid, search_type),
        )
    else:
        await _db.execute(
            """
            UPDATE miner_concurrency
            SET consecutive_failures = ?, updated_at = ?
            WHERE uid = ? AND search_type = ?
            """,
            (new_count, now, uid, search_type),
        )
    await _db.commit()
    return flip


async def get_unreachable_uids(search_type: str) -> set[int]:
    cursor = await _db.execute(
        """
        SELECT uid FROM miner_concurrency
        WHERE search_type = ? AND unreachable_since IS NOT NULL
        """,
        (search_type,),
    )
    return {row["uid"] async for row in cursor}


async def get_unreachable_rows(search_type: str) -> list[dict]:
    cursor = await _db.execute(
        """
        SELECT uid, verified, last_decay_at, unreachable_since
        FROM miner_concurrency
        WHERE search_type = ? AND unreachable_since IS NOT NULL
        """,
        (search_type,),
    )
    return [dict(row) async for row in cursor]


async def apply_decay_tick(
    uid: int, search_type: str, new_verified: int, new_last_decay_at: str
) -> None:
    await _db.execute(
        """
        UPDATE miner_concurrency
        SET verified = ?, last_decay_at = ?, updated_at = ?
        WHERE uid = ? AND search_type = ?
        """,
        (new_verified, new_last_decay_at, _now_iso(), uid, search_type),
    )
    await _db.commit()


async def get_all_rows() -> list[dict]:
    """All miner_concurrency rows across every search type.
    Used by the public API to build the miner list."""
    cursor = await _db.execute("SELECT * FROM miner_concurrency")
    return [dict(row) async for row in cursor]


async def get_rows_for_hotkey(hotkey: str) -> list[dict]:
    cursor = await _db.execute(
        "SELECT * FROM miner_concurrency WHERE hotkey = ?",
        (hotkey,),
    )
    return [dict(row) async for row in cursor]


async def get_windows_for_hotkey(
    hotkey: str, search_type: str, since_hours: int = 72
) -> list[dict]:
    """Scoring windows for ``hotkey`` in ``search_type`` over the last
    ``since_hours``. Filtering on ``hotkey`` (not ``uid``) ensures windows
    from a prior holder of the same UID slot are excluded."""
    cutoff = (datetime.now(timezone.utc) - timedelta(hours=since_hours)).isoformat()
    cursor = await _db.execute(
        """
        SELECT window_start, quality_score, passed, verified_concurrency, created_at
        FROM scoring_windows
        WHERE hotkey = ? AND search_type = ? AND created_at >= ?
        ORDER BY window_start ASC
        """,
        (hotkey, search_type, cutoff),
    )
    return [dict(row) async for row in cursor]
