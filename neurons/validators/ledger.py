"""The verdicts this validator reached itself, from which it sets its own weights."""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path

from desearch.credit import COVERAGE_GATE, SHARE_WINDOW_H


class Ledger:
    def __init__(self, path: Path | str):
        self.db = sqlite3.connect(str(path), check_same_thread=False)
        self.db.executescript(
            """
            CREATE TABLE IF NOT EXISTS verdicts (
                task_id   TEXT PRIMARY KEY,
                kind      TEXT NOT NULL,
                miner     TEXT NOT NULL,
                scored_at REAL NOT NULL,
                verdict   TEXT NOT NULL,
                credited  INTEGER NOT NULL,
                assigned  INTEGER NOT NULL,
                returned  INTEGER NOT NULL
            );
            CREATE INDEX IF NOT EXISTS verdicts_scored ON verdicts (scored_at);
            """
        )
        self.db.commit()

    def record(
        self,
        task_id: str,
        kind: str,
        miner: str,
        verdict: str,
        credited: int,
        assigned: int,
        returned: int,
        at: float | None = None,
    ) -> None:
        self.db.execute(
            "INSERT OR REPLACE INTO verdicts VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                task_id,
                kind,
                miner,
                at or time.time(),
                verdict,
                credited,
                assigned,
                returned,
            ),
        )
        self.db.commit()

    def count(self, now: float | None = None) -> int:
        (count,) = self.db.execute(
            "SELECT COUNT(*) FROM verdicts WHERE scored_at >= ?",
            ((now or time.time()) - SHARE_WINDOW_H * 3600,),
        ).fetchone()
        return count

    def shares(self, now: float | None = None) -> dict[str, dict[str, float]]:
        """Each miner's part of the work this validator verified in the window."""
        since = (now or time.time()) - SHARE_WINDOW_H * 3600
        earned: dict[str, dict[str, int]] = {}
        for kind, miner, credited, assigned, returned in self.db.execute(
            "SELECT kind, miner, SUM(credited), SUM(assigned), SUM(returned)"
            " FROM verdicts WHERE scored_at >= ? GROUP BY kind, miner",
            (since,),
        ):
            if credited <= 0:
                continue
            if kind == "crawl" and assigned and returned / assigned < COVERAGE_GATE:
                continue
            earned.setdefault(kind, {})[miner] = credited
        return {
            kind: {
                miner: amount / sum(miners.values()) for miner, amount in miners.items()
            }
            for kind, miners in earned.items()
        }

    def prune(self, keep_days: int = 7) -> None:
        self.db.execute(
            "DELETE FROM verdicts WHERE scored_at < ?",
            (time.time() - keep_days * 86400,),
        )
        self.db.commit()

    def close(self) -> None:
        self.db.close()
