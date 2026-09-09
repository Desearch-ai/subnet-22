"""Earned concurrency. Stored outside Redis so losing the queue does not reset it."""

from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass

START = 1
CEILING = 30
COVERAGE_GATE = 0.85

# A typed fetch failure is absent on purpose: the miner still made the request.
REWARD = "verified"
PENALTIES = ("lease_expired", "abandoned", "verification_failed")


@dataclass
class Miner:
    hotkey: str
    budget: int
    urls_verified: int
    urls_assigned: int = 0
    urls_returned: int = 0

    @property
    def coverage(self) -> float:
        return self.urls_returned / self.urls_assigned if self.urls_assigned else 1.0


class Budgets:
    def __init__(self, path: str = "budgets.db"):
        self.db = sqlite3.connect(path, check_same_thread=False)
        self.db.executescript(
            """
            CREATE TABLE IF NOT EXISTS miners (
                hotkey        TEXT PRIMARY KEY,
                budget        INTEGER NOT NULL DEFAULT 1,
                urls_verified INTEGER NOT NULL DEFAULT 0,
                urls_assigned INTEGER NOT NULL DEFAULT 0,
                urls_returned INTEGER NOT NULL DEFAULT 0
            );
            CREATE TABLE IF NOT EXISTS budget_events (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                hotkey     TEXT NOT NULL,
                old_budget INTEGER NOT NULL,
                new_budget INTEGER NOT NULL,
                cause      TEXT NOT NULL,
                task_id    TEXT,
                at         REAL NOT NULL
            );
            """
        )
        self.db.commit()

    def get(self, hotkey: str) -> Miner:
        row = self.db.execute(
            "SELECT hotkey, budget, urls_verified, urls_assigned, urls_returned"
            " FROM miners WHERE hotkey = ?",
            (hotkey,),
        ).fetchone()
        if row is None:
            self.db.execute(
                "INSERT INTO miners (hotkey, budget) VALUES (?, ?)", (hotkey, START)
            )
            self.db.commit()
            return Miner(hotkey, START, 0)
        return Miner(*row)

    def assign(self, hotkey: str, urls: int) -> None:
        self.get(hotkey)
        self.db.execute(
            "UPDATE miners SET urls_assigned = urls_assigned + ? WHERE hotkey = ?",
            (urls, hotkey),
        )
        self.db.commit()

    def returned(self, hotkey: str, urls: int) -> None:
        self.db.execute(
            "UPDATE miners SET urls_returned = urls_returned + ? WHERE hotkey = ?",
            (urls, hotkey),
        )
        self.db.commit()

    def _move(self, hotkey: str, new: int, cause: str, task_id: str | None) -> Miner:
        miner = self.get(hotkey)
        new = max(1, min(CEILING, new))
        self.db.execute("UPDATE miners SET budget = ? WHERE hotkey = ?", (new, hotkey))
        self.db.execute(
            "INSERT INTO budget_events (hotkey, old_budget, new_budget, cause, task_id, at)"
            " VALUES (?, ?, ?, ?, ?, ?)",
            (hotkey, miner.budget, new, cause, task_id, time.time()),
        )
        self.db.commit()
        return Miner(
            hotkey, new, miner.urls_verified, miner.urls_assigned, miner.urls_returned
        )

    def reward(self, hotkey: str, task_id: str, urls: int) -> Miner:
        miner = self.get(hotkey)
        self.db.execute(
            "UPDATE miners SET urls_verified = urls_verified + ? WHERE hotkey = ?",
            (urls, hotkey),
        )
        return self._move(hotkey, miner.budget + 1, REWARD, task_id)

    def penalise(self, hotkey: str, task_id: str, cause: str) -> Miner:
        assert cause in PENALTIES, cause
        return self._move(hotkey, self.get(hotkey).budget // 2, cause, task_id)

    def history(self, hotkey: str) -> list[dict]:
        rows = self.db.execute(
            "SELECT old_budget, new_budget, cause, task_id, at FROM budget_events"
            " WHERE hotkey = ? ORDER BY id",
            (hotkey,),
        ).fetchall()
        return [
            {"old": r[0], "new": r[1], "cause": r[2], "task_id": r[3], "at": r[4]}
            for r in rows
        ]

    def shares(self) -> dict[str, float]:
        rows = self.db.execute(
            "SELECT hotkey, urls_verified, urls_assigned, urls_returned FROM miners"
        ).fetchall()
        eligible = [
            (hotkey, verified)
            for hotkey, verified, assigned, returned in rows
            if verified > 0 and (returned / assigned if assigned else 1.0) >= COVERAGE_GATE
        ]
        total = sum(verified for _, verified in eligible)
        return {hotkey: verified / total for hotkey, verified in eligible} if total else {}

    def coverage_report(self) -> dict[str, dict]:
        rows = self.db.execute(
            "SELECT hotkey, urls_assigned, urls_returned FROM miners WHERE urls_assigned > 0"
        ).fetchall()
        out = {}
        for hotkey, assigned, returned in rows:
            coverage = returned / assigned if assigned else 1.0
            out[hotkey] = {
                "assigned": assigned,
                "returned": returned,
                "coverage": round(coverage, 4),
                "eligible": coverage >= COVERAGE_GATE,
            }
        return out
