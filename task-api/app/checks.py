"""What validators' checks found about each hotkey: its passes, its recent fails and its re-checks."""

from __future__ import annotations

import sqlite3
import time

from desearch.credit import RATE_WINDOW_S, RECENT_CHECKS

from .sampling import RECHECK_UPLOADS

KEEP_DAYS = 30
DAY = 86400


class Checks:
    def __init__(self, db: sqlite3.Connection):
        self.db = db
        self.db.executescript(
            """
            CREATE TABLE IF NOT EXISTS checks (
                id                 INTEGER PRIMARY KEY AUTOINCREMENT,
                hotkey             TEXT NOT NULL,
                task_id            TEXT NOT NULL,
                at                 REAL NOT NULL,
                upload_at          REAL NOT NULL,
                passed             INTEGER NOT NULL,
                errors_judged      INTEGER NOT NULL DEFAULT 0,
                errors_unconfirmed INTEGER NOT NULL DEFAULT 0
            );
            CREATE INDEX IF NOT EXISTS checks_hotkey ON checks (hotkey, at);
            CREATE TABLE IF NOT EXISTS check_state (
                hotkey       TEXT PRIMARY KEY,
                recheck_left INTEGER NOT NULL DEFAULT 0,
                last_pass_at REAL NOT NULL DEFAULT 0
            );
            """
        )
        self.db.commit()

    def record(
        self,
        hotkey: str,
        task_id: str,
        passed: bool,
        upload_at: float,
        errors_judged: int = 0,
        errors_unconfirmed: int = 0,
        at: float | None = None,
    ) -> None:
        self.db.execute(
            "INSERT INTO checks (hotkey, task_id, at, upload_at, passed, errors_judged,"
            " errors_unconfirmed) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                hotkey,
                task_id,
                at or time.time(),
                upload_at,
                int(passed),
                errors_judged,
                errors_unconfirmed,
            ),
        )
        if passed:
            self.db.execute(
                "INSERT INTO check_state (hotkey, last_pass_at) VALUES (?, ?)"
                " ON CONFLICT (hotkey) DO UPDATE SET"
                " last_pass_at = MAX(last_pass_at, excluded.last_pass_at)",
                (hotkey, upload_at),
            )
        self.db.commit()

    def passes(self, hotkey: str) -> int:
        (count,) = self.db.execute(
            "SELECT COUNT(*) FROM checks WHERE hotkey = ? AND passed = 1", (hotkey,)
        ).fetchone()
        return count

    def fails_in_recent(self, hotkey: str, count: int = RECENT_CHECKS) -> int:
        rows = self.db.execute(
            "SELECT passed FROM checks WHERE hotkey = ? ORDER BY at DESC, id DESC LIMIT ?",
            (hotkey, count),
        ).fetchall()
        return sum(1 for (passed,) in rows if not passed)

    def last_pass_at(self, hotkey: str) -> float:
        found = self.db.execute(
            "SELECT last_pass_at FROM check_state WHERE hotkey = ?", (hotkey,)
        ).fetchone()
        return found[0] if found else 0.0

    def recheck_left(self, hotkey: str) -> int:
        found = self.db.execute(
            "SELECT recheck_left FROM check_state WHERE hotkey = ?", (hotkey,)
        ).fetchone()
        return found[0] if found else 0

    def start_recheck(self, hotkey: str, uploads: int = RECHECK_UPLOADS) -> None:
        self.db.execute(
            "INSERT INTO check_state (hotkey, recheck_left) VALUES (?, ?)"
            " ON CONFLICT (hotkey) DO UPDATE SET recheck_left = excluded.recheck_left",
            (hotkey, uploads),
        )
        self.db.commit()

    def took_recheck(self, hotkey: str) -> None:
        self.db.execute(
            "UPDATE check_state SET recheck_left = MAX(recheck_left - 1, 0) WHERE hotkey = ?",
            (hotkey,),
        )
        self.db.commit()

    def error_share(self, hotkey: str, now: float | None = None) -> float:
        """Share of the hotkey's reported failures that checks reproduced, one of each assumed before any."""
        judged, unconfirmed = self.db.execute(
            "SELECT COALESCE(SUM(errors_judged), 0), COALESCE(SUM(errors_unconfirmed), 0)"
            " FROM checks WHERE hotkey = ? AND at >= ?",
            (hotkey, (now or time.time()) - RATE_WINDOW_S),
        ).fetchone()
        return (judged - unconfirmed + 1) / (judged + 2)

    def prune(self, keep_days: int = KEEP_DAYS) -> None:
        self.db.execute(
            "DELETE FROM checks WHERE at < ?", (time.time() - keep_days * DAY,)
        )
        self.db.commit()
