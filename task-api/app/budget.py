from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass

START = 1
CEILING = 30
COVERAGE_GATE = 0.85
CRAWL = "crawl"
SHARE_WINDOW_H = 24
KEEP_H = SHARE_WINDOW_H * 7
HOUR = 3600

# Fetch failures are left out on purpose: the miner still tried.
REWARD = "verified"
PENALTIES = ("lease_expired", "abandoned", "verification_failed")

# Only fails the miner caused count; "unscorable" is the validator's own timeout or crash.
STRIKE_REASONS = frozenset(
    {
        "extra_rows",
        "coverage",
        "text_not_from_html",
        "content_mismatch",
        "errors_not_reproducible",
        "unreadable",
    }
)
# Both must hold: a busy honest miner meets two bad batches a day, a cheater fails most tasks.
STRIKES_TO_LOCK = 2
STRIKE_SHARE = 0.05
STRIKE_WINDOW_H = 24
LOCKOUT_H = 12


@dataclass
class MinerBudget:
    hotkey: str
    budget: int
    urls_verified: int


def hour_of(at: float | None = None) -> int:
    return int((at or time.time()) // HOUR)


class Budgets:
    def __init__(self, db: sqlite3.Connection):
        self.db = db
        self.db.executescript(
            """
            CREATE TABLE IF NOT EXISTS miners (
                hotkey        TEXT PRIMARY KEY,
                budget        INTEGER NOT NULL DEFAULT 1,
                urls_verified INTEGER NOT NULL DEFAULT 0
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
            CREATE INDEX IF NOT EXISTS budget_events_hotkey ON budget_events (hotkey, id);
            CREATE TABLE IF NOT EXISTS credits (
                pool   TEXT NOT NULL,
                hotkey TEXT NOT NULL,
                hour   INTEGER NOT NULL,
                amount INTEGER NOT NULL DEFAULT 0,
                PRIMARY KEY (pool, hotkey, hour)
            );
            CREATE INDEX IF NOT EXISTS credits_hour ON credits (hour);
            CREATE TABLE IF NOT EXISTS coverage (
                hotkey   TEXT NOT NULL,
                hour     INTEGER NOT NULL,
                assigned INTEGER NOT NULL DEFAULT 0,
                returned INTEGER NOT NULL DEFAULT 0,
                PRIMARY KEY (hotkey, hour)
            );
            CREATE INDEX IF NOT EXISTS coverage_hour ON coverage (hour);
            CREATE TABLE IF NOT EXISTS strikes (
                id      INTEGER PRIMARY KEY AUTOINCREMENT,
                hotkey  TEXT NOT NULL,
                reason  TEXT NOT NULL,
                task_id TEXT,
                at      REAL NOT NULL
            );
            CREATE INDEX IF NOT EXISTS strikes_hotkey ON strikes (hotkey, at);
            CREATE TABLE IF NOT EXISTS lockouts (
                hotkey TEXT PRIMARY KEY,
                until  REAL NOT NULL,
                reason TEXT NOT NULL
            );
            """
        )
        self.db.commit()

    def get_or_create(self, hotkey: str) -> MinerBudget:
        miner = self.get(hotkey)
        self.db.execute(
            "INSERT OR IGNORE INTO miners (hotkey, budget) VALUES (?, ?)",
            (hotkey, START),
        )
        self.db.commit()
        return miner

    def get(self, hotkey: str) -> MinerBudget:
        row = self.db.execute(
            "SELECT hotkey, budget, urls_verified FROM miners WHERE hotkey = ?",
            (hotkey,),
        ).fetchone()
        return MinerBudget(*row) if row else MinerBudget(hotkey, START, 0)

    def record_coverage(self, hotkey: str, assigned: int, returned: int) -> None:
        """In-flight work never counts toward coverage."""
        self.db.execute(
            "INSERT INTO coverage (hotkey, hour, assigned, returned) VALUES (?, ?, ?, ?)"
            " ON CONFLICT (hotkey, hour) DO UPDATE SET assigned = assigned + excluded.assigned,"
            " returned = returned + excluded.returned",
            (hotkey, hour_of(), assigned, min(returned, assigned)),
        )
        self.db.commit()

    def _set_budget(
        self, hotkey: str, new: int, cause: str, task_id: str | None
    ) -> MinerBudget:
        miner = self.get_or_create(hotkey)
        new = max(1, min(CEILING, new))
        self.db.execute("UPDATE miners SET budget = ? WHERE hotkey = ?", (new, hotkey))
        self.db.execute(
            "INSERT INTO budget_events (hotkey, old_budget, new_budget, cause, task_id, at)"
            " VALUES (?, ?, ?, ?, ?, ?)",
            (hotkey, miner.budget, new, cause, task_id, time.time()),
        )
        self.db.commit()
        return MinerBudget(hotkey, new, miner.urls_verified)

    def reward(
        self,
        hotkey: str,
        task_id: str,
        amount: int,
        ramp: bool = True,
        pool: str = CRAWL,
    ) -> MinerBudget:
        miner = self.get_or_create(hotkey)
        if pool == CRAWL:
            self.db.execute(
                "UPDATE miners SET urls_verified = urls_verified + ? WHERE hotkey = ?",
                (amount, hotkey),
            )
        self.db.execute(
            "INSERT INTO credits (pool, hotkey, hour, amount) VALUES (?, ?, ?, ?)"
            " ON CONFLICT (pool, hotkey, hour) DO UPDATE SET amount = amount + excluded.amount",
            (pool, hotkey, hour_of(), amount),
        )
        if not ramp:
            self.db.commit()
            return self.get_or_create(hotkey)
        return self._set_budget(hotkey, miner.budget + 1, REWARD, task_id)

    def penalise(self, hotkey: str, task_id: str, cause: str) -> MinerBudget:
        assert cause in PENALTIES, cause
        return self._set_budget(
            hotkey, self.get_or_create(hotkey).budget // 2, cause, task_id
        )

    def strike(
        self,
        hotkey: str,
        reason: str,
        task_id: str,
        judged: int,
        now: float | None = None,
    ) -> float | None:
        """The lockout's end if this strike, among `judged` recent verdicts, starts one."""
        now = now or time.time()
        self.db.execute(
            "INSERT INTO strikes (hotkey, reason, task_id, at) VALUES (?, ?, ?, ?)",
            (hotkey, reason, task_id, now),
        )
        (recent,) = self.db.execute(
            "SELECT COUNT(*) FROM strikes WHERE hotkey = ? AND at > ?",
            (hotkey, now - STRIKE_WINDOW_H * HOUR),
        ).fetchone()
        until = None
        if recent >= STRIKES_TO_LOCK and recent >= STRIKE_SHARE * judged:
            until = now + LOCKOUT_H * HOUR
            self.db.execute(
                "INSERT INTO lockouts (hotkey, until, reason) VALUES (?, ?, ?)"
                " ON CONFLICT (hotkey) DO UPDATE SET until = excluded.until,"
                " reason = excluded.reason",
                (hotkey, until, reason),
            )
        self.db.commit()
        return until

    def locked_until(self, hotkey: str, now: float | None = None) -> float | None:
        row = self.db.execute(
            "SELECT until FROM lockouts WHERE hotkey = ?", (hotkey,)
        ).fetchone()
        return row[0] if row and row[0] > (now or time.time()) else None

    def history(self, hotkey: str, limit: int = 100) -> list[dict]:
        rows = self.db.execute(
            "SELECT old_budget, new_budget, cause, task_id, at FROM budget_events"
            " WHERE hotkey = ? ORDER BY id DESC LIMIT ?",
            (hotkey, limit),
        ).fetchall()
        return [
            {"old": r[0], "new": r[1], "cause": r[2], "task_id": r[3], "at": r[4]}
            for r in rows
        ]

    def shares(
        self, window_hours: int = SHARE_WINDOW_H, now: float | None = None
    ) -> dict[str, dict[str, float]]:
        since = hour_of(now) - window_hours + 1
        covered = self.coverage_report(window_hours, now)
        earned: dict[str, dict[str, int]] = {}
        for pool, hotkey, amount in self.db.execute(
            "SELECT pool, hotkey, SUM(amount) FROM credits WHERE hour >= ?"
            " GROUP BY pool, hotkey",
            (since,),
        ):
            if amount <= 0:
                continue
            if pool == CRAWL and not covered.get(hotkey, {}).get("eligible", True):
                continue
            earned.setdefault(pool, {})[hotkey] = amount
        return {
            pool: {
                hotkey: amount / sum(miners.values())
                for hotkey, amount in miners.items()
            }
            for pool, miners in earned.items()
        }

    def coverage_report(
        self, window_hours: int = SHARE_WINDOW_H, now: float | None = None
    ) -> dict[str, dict]:
        since = hour_of(now) - window_hours + 1
        rows = self.db.execute(
            "SELECT hotkey, SUM(assigned), SUM(returned) FROM coverage"
            " WHERE hour >= ? GROUP BY hotkey HAVING SUM(assigned) > 0",
            (since,),
        ).fetchall()
        return {
            hotkey: {
                "assigned": assigned,
                "returned": returned,
                "coverage": round(returned / assigned, 4),
                "eligible": returned / assigned >= COVERAGE_GATE,
            }
            for hotkey, assigned, returned in rows
        }

    def prune(self, keep_hours: int = KEEP_H) -> None:
        oldest = hour_of() - keep_hours
        self.db.execute("DELETE FROM credits WHERE hour < ?", (oldest,))
        self.db.execute("DELETE FROM coverage WHERE hour < ?", (oldest,))
        self.db.execute("DELETE FROM strikes WHERE at < ?", (oldest * HOUR,))
        self.db.commit()
