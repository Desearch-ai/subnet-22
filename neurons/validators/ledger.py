"""The verdicts this validator reached itself, from which it sets its own weights."""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path

from desearch.credit import SHARE_WINDOW_H
from desearch.kinds import CRAWL


TRUSTED_PASSES = 3
# A failed crawl task takes its assigned URLs back, so failing costs more than not trying.
NET = f"SUM(CASE WHEN kind = '{CRAWL}' AND verdict = 'fail' THEN -assigned ELSE credited END)"


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

    def window(self, now: float | None = None) -> list[dict]:
        """Per miner and kind, what this validator checked inside the scoring window."""
        since = (now or time.time()) - SHARE_WINDOW_H * 3600
        rows = self.db.execute(
            "SELECT kind, miner, COUNT(*), SUM(verdict = 'pass'), SUM(verdict = 'fail'),"
            f" SUM(assigned), SUM(returned), SUM(credited), {NET} FROM verdicts"
            " WHERE scored_at >= ? GROUP BY kind, miner ORDER BY SUM(credited) DESC",
            (since,),
        )
        return [
            {
                "kind": kind,
                "miner": miner,
                "tasks": tasks,
                "passed": passed,
                "failed": failed,
                "assigned": assigned,
                "returned": returned,
                "credited": credited,
                "net": net,
            }
            for kind, miner, tasks, passed, failed, assigned, returned, credited, net in rows
        ]

    def shares(self, now: float | None = None) -> dict[str, dict[str, float]]:
        """Each miner's part of the work this validator verified, less its failed tasks."""
        since = (now or time.time()) - SHARE_WINDOW_H * 3600
        earned: dict[str, dict[str, int]] = {}
        for kind, miner, net in self.db.execute(
            f"SELECT kind, miner, {NET} FROM verdicts WHERE scored_at >= ?"
            " GROUP BY kind, miner",
            (since,),
        ):
            if net > 0:
                earned.setdefault(kind, {})[miner] = net
        return {
            kind: {
                miner: amount / sum(miners.values()) for miner, amount in miners.items()
            }
            for kind, miners in earned.items()
        }

    def trusted(self, miner: str, now: float | None = None) -> bool:
        """A few passes behind it and no fail in the last day."""
        now = now or time.time()
        passed, failed = self.db.execute(
            "SELECT COALESCE(SUM(verdict = 'pass'), 0),"
            " COALESCE(SUM(verdict = 'fail' AND scored_at >= ?), 0)"
            " FROM verdicts WHERE miner = ?",
            (now - SHARE_WINDOW_H * 3600, miner),
        ).fetchone()
        return passed >= TRUSTED_PASSES and not failed

    def prune(self, keep_days: int = 7) -> None:
        self.db.execute(
            "DELETE FROM verdicts WHERE scored_at < ?",
            (time.time() - keep_days * 86400,),
        )
        self.db.commit()

    def close(self) -> None:
        self.db.close()
