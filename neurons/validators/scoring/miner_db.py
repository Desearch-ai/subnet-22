"""
SQLite persistence for miner concurrency state and scoring window history.

The writer process (neuron) holds a single long-lived connection; reader
processes (public API) open a short-lived connection per call so no reader
mark blocks SQLite's WAL file from resetting.
"""

from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from typing import Optional

import aiosqlite
import bittensor as bt

_writer_db: Optional[aiosqlite.Connection] = None
_readonly_path: Optional[str] = None

STALENESS_HOURS = 24
RETENTION_DAYS = 3
PUBLIC_API_VISIBILITY_HOURS = 4

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


@asynccontextmanager
async def _conn():
    """Yield a connection for the current mode. Writer yields the shared
    long-lived connection; reader opens a short-lived per-call connection so
    no reader mark blocks the WAL from resetting."""
    if _writer_db is not None:
        yield _writer_db
        return
    if _readonly_path is None:
        raise RuntimeError("miner_db not initialized")
    uri = f"file:{_readonly_path}?mode=ro"
    conn = await aiosqlite.connect(uri, uri=True)
    try:
        conn.row_factory = aiosqlite.Row
        yield conn
    finally:
        await conn.close()


async def initialize(db_path: str, readonly: bool = False) -> None:
    """Configure the module for ``db_path``.

    ``readonly=True`` just stores the path — each read opens a fresh
    connection via ``file:…?mode=ro`` so a public-API process cannot mutate
    state and does not hold a long-lived reader that would block WAL reset.

    The writer process calls this with ``readonly=False``, opens the shared
    long-lived connection, creates the schema, drains any accumulated WAL
    from prior runs, and runs retention cleanup."""
    global _writer_db, _readonly_path
    if readonly:
        _readonly_path = db_path
        bt.logging.info(f"[MinerDB] Configured read-only at {db_path}")
        return

    _writer_db = await aiosqlite.connect(db_path)
    _writer_db.row_factory = aiosqlite.Row
    await _writer_db.execute("PRAGMA journal_mode=WAL")

    for statement in _SCHEMA.strip().split(";"):
        statement = statement.strip()
        if statement:
            await _writer_db.execute(statement)
    await _writer_db.commit()

    # Drain any WAL accumulated from prior runs where long-held readers
    # prevented auto-checkpoint from resetting the file.
    await _writer_db.execute("PRAGMA wal_checkpoint(TRUNCATE)")

    await _purge_old_history()
    await _decay_stale_verified()

    bt.logging.info(f"[MinerDB] Initialized at {db_path}")


async def close() -> None:
    global _writer_db, _readonly_path
    if _writer_db is not None:
        await _writer_db.close()
        _writer_db = None
    _readonly_path = None


async def _purge_old_history() -> None:
    cutoff = (datetime.now(timezone.utc) - timedelta(days=RETENTION_DAYS)).isoformat()
    async with _conn() as db:
        await db.execute("DELETE FROM scoring_windows WHERE created_at < ?", (cutoff,))
        await db.commit()


async def _decay_stale_verified() -> None:
    cutoff = (datetime.now(timezone.utc) - timedelta(hours=STALENESS_HOURS)).isoformat()
    async with _conn() as db:
        result = await db.execute(
            "UPDATE miner_concurrency SET verified = 1 WHERE updated_at < ?", (cutoff,)
        )
        if result.rowcount:
            bt.logging.info(
                f"[MinerDB] Decayed {result.rowcount} stale miners to verified=1"
            )
        await db.commit()


async def get_verified(uid: int, search_type: str) -> int:
    async with _conn() as db:
        cursor = await db.execute(
            "SELECT verified FROM miner_concurrency WHERE uid = ? AND search_type = ?",
            (uid, search_type),
        )
        row = await cursor.fetchone()
    return row["verified"] if row else 1


async def get_all_verified(search_type: str) -> dict[int, int]:
    async with _conn() as db:
        cursor = await db.execute(
            "SELECT uid, verified FROM miner_concurrency WHERE search_type = ?",
            (search_type,),
        )
        return {row["uid"]: row["verified"] async for row in cursor}


async def get_all_concurrency_data(
    search_type: str,
) -> dict[int, tuple[float, int]]:
    """Return {uid: (quality_avg, verified)} for all miners of a given search type."""
    async with _conn() as db:
        cursor = await db.execute(
            """
            SELECT uid, quality_avg, verified
            FROM miner_concurrency WHERE search_type = ?
            """,
            (search_type,),
        )
        return {
            row["uid"]: (row["quality_avg"], row["verified"]) async for row in cursor
        }


async def get_concurrency_row(uid: int, search_type: str) -> Optional[dict]:
    async with _conn() as db:
        cursor = await db.execute(
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
    async with _conn() as db:
        await db.execute(
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
        await db.commit()


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
    async with _conn() as db:
        await db.execute(
            """
            DELETE FROM miner_concurrency
            WHERE uid = ? AND search_type = ? AND hotkey != ?
            """,
            (uid, search_type, hotkey),
        )
        await db.execute(
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
        await db.commit()


async def promote_pending_declared() -> int:
    async with _conn() as db:
        cursor = await db.execute(
            """
            UPDATE miner_concurrency
            SET declared = pending_declared,
                pending_declared = NULL,
                updated_at = ?
            WHERE pending_declared IS NOT NULL
            """,
            (_now_iso(),),
        )
        await db.commit()
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
    async with _conn() as db:
        await db.execute(
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
        await db.commit()


async def count_failed_windows(
    uid: int, search_type: str, since_hours: int = 12
) -> int:
    cutoff = (datetime.now(timezone.utc) - timedelta(hours=since_hours)).isoformat()
    async with _conn() as db:
        cursor = await db.execute(
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
    async with _conn() as db:
        cursor = await db.execute(
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
        await db.execute(
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
        await db.commit()
    return was_unreachable


async def record_call_failure(uid: int, search_type: str, threshold: int) -> bool:
    """Increment ``consecutive_failures`` and mark unreachable when the counter
    crosses ``threshold`` for the first time. Returns ``True`` on that
    transition. No-op (returns ``False``) if the row doesn't exist."""
    async with _conn() as db:
        cursor = await db.execute(
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
            await db.execute(
                """
                UPDATE miner_concurrency
                SET consecutive_failures = ?,
                    unreachable_since = ?,
                    last_decay_at = ?
                WHERE uid = ? AND search_type = ?
                """,
                (new_count, now, now, uid, search_type),
            )
        else:
            await db.execute(
                """
                UPDATE miner_concurrency
                SET consecutive_failures = ?
                WHERE uid = ? AND search_type = ?
                """,
                (new_count, uid, search_type),
            )
        await db.commit()
    return flip


async def get_unreachable_uids(search_type: str) -> set[int]:
    async with _conn() as db:
        cursor = await db.execute(
            """
            SELECT uid FROM miner_concurrency
            WHERE search_type = ? AND unreachable_since IS NOT NULL
            """,
            (search_type,),
        )
        return {row["uid"] async for row in cursor}


async def get_unreachable_rows(search_type: str) -> list[dict]:
    async with _conn() as db:
        cursor = await db.execute(
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
    async with _conn() as db:
        await db.execute(
            """
            UPDATE miner_concurrency
            SET verified = ?, last_decay_at = ?
            WHERE uid = ? AND search_type = ?
            """,
            (new_verified, new_last_decay_at, uid, search_type),
        )
        await db.commit()


async def get_all_rows() -> list[dict]:
    """Rows for miners the public API still surfaces: last confirmed alive
    within the last ``PUBLIC_API_VISIBILITY_HOURS`` hours. Unreachable rows
    stay visible (with ``unreachable_since`` set) so the UI can render them
    in an "Unreachable" bucket; only rows whose ``updated_at`` has gone
    cold past the cutoff drop out entirely. ``updated_at`` is refreshed only
    on positive signals (successful IsAlive, recovery, scoring) — failure
    and decay writes leave it alone so the field tracks last-seen-alive."""
    cutoff = (
        datetime.now(timezone.utc) - timedelta(hours=PUBLIC_API_VISIBILITY_HOURS)
    ).isoformat()
    async with _conn() as db:
        cursor = await db.execute(
            "SELECT * FROM miner_concurrency WHERE updated_at > ?",
            (cutoff,),
        )
        return [dict(row) async for row in cursor]


async def get_rows_for_hotkey(hotkey: str) -> list[dict]:
    cutoff = (
        datetime.now(timezone.utc) - timedelta(hours=PUBLIC_API_VISIBILITY_HOURS)
    ).isoformat()
    async with _conn() as db:
        cursor = await db.execute(
            "SELECT * FROM miner_concurrency WHERE hotkey = ? AND updated_at > ?",
            (hotkey, cutoff),
        )
        return [dict(row) async for row in cursor]


async def get_windows_for_hotkey(
    hotkey: str, search_type: str, since_hours: int = 72
) -> list[dict]:
    """Scoring windows for ``hotkey`` in ``search_type`` over the last
    ``since_hours``. Filtering on ``hotkey`` (not ``uid``) ensures windows
    from a prior holder of the same UID slot are excluded."""
    cutoff = (datetime.now(timezone.utc) - timedelta(hours=since_hours)).isoformat()
    async with _conn() as db:
        cursor = await db.execute(
            """
            SELECT window_start, quality_score, passed, verified_concurrency, created_at
            FROM scoring_windows
            WHERE hotkey = ? AND search_type = ? AND created_at >= ?
            ORDER BY window_start ASC
            """,
            (hotkey, search_type, cutoff),
        )
        return [dict(row) async for row in cursor]
