"""The verdicts this validator reached itself and every miner's logged uploads, from which it sets its own weights."""

from __future__ import annotations

import sqlite3
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from desearch.credit import (
    FAILS_FOR_PENALTY,
    PENALTY_WINDOW_S,
    RATE_WINDOW_S,
    RECENT_CHECKS,
    SHARE_WINDOW_H,
    overstated,
)
from desearch.kinds import CRAWL


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
            CREATE TABLE IF NOT EXISTS uploads (
                key          TEXT PRIMARY KEY,
                task_id      TEXT NOT NULL,
                miner        TEXT NOT NULL,
                completed_at REAL NOT NULL,
                assigned     INTEGER NOT NULL,
                ok           INTEGER NOT NULL,
                errors       INTEGER NOT NULL
            );
            CREATE INDEX IF NOT EXISTS uploads_completed ON uploads (completed_at);
            CREATE TABLE IF NOT EXISTS checks (
                key          TEXT PRIMARY KEY,
                miner        TEXT NOT NULL,
                completed_at REAL NOT NULL,
                scored_at    REAL NOT NULL,
                verdict      TEXT NOT NULL,
                credited     INTEGER NOT NULL,
                returned     INTEGER NOT NULL,
                content      INTEGER NOT NULL,
                assigned     INTEGER NOT NULL
            );
            CREATE INDEX IF NOT EXISTS checks_scored ON checks (scored_at);
            CREATE TABLE IF NOT EXISTS state (
                name  TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
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

    def add_uploads(self, entries: list[dict]) -> int:
        before = self.db.total_changes
        self.db.executemany(
            "INSERT OR IGNORE INTO uploads VALUES (?, ?, ?, ?, ?, ?, ?)",
            [
                (
                    e["key"],
                    e["task_id"],
                    e["miner"],
                    e["completed_at"],
                    e["assigned"],
                    e["ok"],
                    e["errors"],
                )
                for e in entries
            ],
        )
        self.db.commit()
        return self.db.total_changes - before

    def record_check(
        self,
        key: str,
        miner: str,
        completed_at: float,
        verdict: str,
        credited: int,
        returned: int,
        content: int,
        assigned: int,
        at: float | None = None,
    ) -> None:
        self.db.execute(
            "INSERT OR REPLACE INTO checks VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                key,
                miner,
                completed_at,
                at or time.time(),
                verdict,
                credited,
                returned,
                content,
                assigned,
            ),
        )
        self.db.commit()

    def state(self, name: str) -> str | None:
        found = self.db.execute(
            "SELECT value FROM state WHERE name = ?", (name,)
        ).fetchone()
        return found[0] if found else None

    def set_state(self, name: str, value: str) -> None:
        self.db.execute("INSERT OR REPLACE INTO state VALUES (?, ?)", (name, value))
        self.db.commit()

    def crawl_paid(self, now: float | None = None) -> dict[str, Paid]:
        """Each miner's rows of the scoring window as this validator's own checks judge them."""
        now = now or time.time()
        since = now - SHARE_WINDOW_H * 3600
        checks: dict[str, list[Check]] = defaultdict(list)
        for (
            miner,
            key,
            completed_at,
            verdict,
            credited,
            returned,
            content,
            assigned,
            ok,
        ) in self.db.execute(
            "SELECT c.miner, c.key, c.completed_at, c.verdict, c.credited, c.returned,"
            " c.content, c.assigned, u.ok FROM checks c LEFT JOIN uploads u ON u.key = c.key"
            " WHERE c.scored_at >= ? AND c.verdict IN ('pass', 'fail') ORDER BY c.scored_at",
            (now - RATE_WINDOW_S,),
        ):
            passed = verdict == "pass" and not (
                ok is not None and overstated(ok, content, assigned)
            )
            checks[miner].append(
                Check(
                    key,
                    completed_at,
                    passed,
                    credited if passed else 0,
                    returned,
                    assigned,
                )
            )
        uploads: dict[str, list[tuple[str, float, int]]] = defaultdict(list)
        for miner, key, completed_at, returned in self.db.execute(
            "SELECT miner, key, completed_at, MIN(ok + errors, assigned) FROM uploads"
            " WHERE completed_at >= ?",
            (since,),
        ):
            uploads[miner].append((key, completed_at, returned))
        return {
            miner: paid(checks[miner], uploads[miner], since)
            for miner in set(checks) | set(uploads)
        }

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

    def prune(self, keep_days: int = 7) -> None:
        oldest = time.time() - keep_days * 86400
        self.db.execute("DELETE FROM verdicts WHERE scored_at < ?", (oldest,))
        self.db.execute("DELETE FROM checks WHERE scored_at < ?", (oldest,))
        self.db.execute("DELETE FROM uploads WHERE completed_at < ?", (oldest,))
        self.db.commit()

    def close(self) -> None:
        self.db.close()


@dataclass
class Check:
    key: str
    completed_at: float
    passed: bool
    credited: int
    returned: int
    assigned: int


@dataclass
class Paid:
    rows: float
    uploads: int
    checked: int
    failed: int
    rate: float


def paid(
    checks: list[Check], uploads: list[tuple[str, float, int]], since: float
) -> Paid:
    """Checked uploads at what they paid, the others at the rate the checks paid, none of what a failed check takes back."""
    judged = sum(check.returned for check in checks)
    rate = sum(check.credited for check in checks) / judged if judged else 0.0
    taken = taken_back(checks)
    checked = {check.key for check in checks}
    recent = [check for check in checks if check.completed_at >= since]
    rows = 0.0
    for check in recent:
        if not check.passed:
            rows -= check.assigned
        elif not taken_at(taken, check.completed_at):
            rows += check.credited
    for key, completed_at, returned in uploads:
        if key not in checked and not taken_at(taken, completed_at):
            rows += rate * returned
    failed = sum(1 for check in recent if not check.passed)
    return Paid(rows, len(uploads), len(recent), failed, round(rate, 4))


def taken_back(checks: list[Check]) -> list[tuple[float, float]]:
    """Upload times a failed check takes back: since the last passed check before it, and a day before a second fail among the last ten."""
    spans, last_pass = [], float("-inf")
    for check in sorted(checks, key=lambda c: c.completed_at):
        if check.passed:
            last_pass = max(last_pass, check.completed_at)
        else:
            spans.append((last_pass, check.completed_at))
    fails = [check for check in checks[-RECENT_CHECKS:] if not check.passed]
    if len(fails) >= FAILS_FOR_PENALTY:
        latest = max(check.completed_at for check in fails)
        spans.append((latest - PENALTY_WINDOW_S, latest))
    return spans


def taken_at(spans: list[tuple[float, float]], at: float) -> bool:
    return any(start < at <= end for start, end in spans)
