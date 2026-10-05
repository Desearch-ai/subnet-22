from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass

START = 1
CEILING = 100
GROWTH = 0.5
WAITING_PER_BUDGET = 2
COVERAGE_GATE = 0.85
CRAWL = "crawl"
EMBED = "embed"
SHARE_WINDOW_H = 24
KEEP_H = SHARE_WINDOW_H * 7
HOUR = 3600

# Fetch failures are left out on purpose: the miner still tried.
REWARD = "verified"
PENALTIES = ("claim_expired", "abandoned", "verification_failed")

# A crash or a short outage lets several claims lapse at once; together they are one strike.
EXPIRY_REASONS = frozenset({"claim_expired", "abandoned"})
STRIKE_BURST_S = 300
# Only fails the miner caused count; "unscorable" is the validator's own timeout or crash.
STRIKE_REASONS = EXPIRY_REASONS | frozenset(
    {
        "extra_rows",
        "coverage",
        "text_not_from_html",
        "content_mismatch",
        "errors_not_reproducible",
        "reported_rows",
        "unreadable",
        "vectors_missing",
        "vectors_malformed",
        "vectors_mismatch",
    }
)
STRIKES_TO_LOCK = 2
STRIKE_SHARE = 0.05
STRIKE_WINDOW_H = 24
LOCKOUT_STEPS_H = (1, 12, 48)
LOCKOUT_MEMORY_H = 7 * 24
HOSTILE_LOCKOUT_H = 7 * 24
FULL_PENALTY_LOCKOUT_H = 48


@dataclass
class MinerBudget:
    hotkey: str
    pool: str
    budget: int
    verified: int


def hour_of(at: float | None = None) -> int:
    return int((at or time.time()) // HOUR)


class Budgets:
    def __init__(self, db: sqlite3.Connection):
        self.db = db
        self.db.executescript(
            """
            CREATE TABLE IF NOT EXISTS miners (
                hotkey   TEXT NOT NULL,
                pool     TEXT NOT NULL,
                budget   INTEGER NOT NULL DEFAULT 1,
                verified INTEGER NOT NULL DEFAULT 0,
                PRIMARY KEY (hotkey, pool)
            );
            CREATE TABLE IF NOT EXISTS budget_events (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                hotkey     TEXT NOT NULL,
                pool       TEXT NOT NULL,
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
                pool    TEXT NOT NULL,
                reason  TEXT NOT NULL,
                task_id TEXT,
                at      REAL NOT NULL
            );
            CREATE INDEX IF NOT EXISTS strikes_hotkey ON strikes (hotkey, pool, at);
            CREATE TABLE IF NOT EXISTS lockouts (
                hotkey TEXT NOT NULL,
                pool   TEXT NOT NULL,
                until  REAL NOT NULL,
                reason TEXT NOT NULL,
                PRIMARY KEY (hotkey, pool)
            );
            CREATE TABLE IF NOT EXISTS lockout_history (
                hotkey TEXT NOT NULL,
                pool   TEXT NOT NULL,
                at     REAL NOT NULL
            );
            CREATE INDEX IF NOT EXISTS lockout_history_hotkey ON lockout_history (hotkey, pool, at);
            """
        )
        self.db.commit()

    def get_or_create(self, hotkey: str, pool: str = CRAWL) -> MinerBudget:
        miner = self.get(hotkey, pool)
        self.db.execute(
            "INSERT OR IGNORE INTO miners (hotkey, pool, budget) VALUES (?, ?, ?)",
            (hotkey, pool, START),
        )
        self.db.commit()
        return miner

    def get(self, hotkey: str, pool: str = CRAWL) -> MinerBudget:
        row = self.db.execute(
            "SELECT hotkey, pool, budget, verified FROM miners"
            " WHERE hotkey = ? AND pool = ?",
            (hotkey, pool),
        ).fetchone()
        return MinerBudget(*row) if row else MinerBudget(hotkey, pool, START, 0)

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
        self, hotkey: str, pool: str, new: int, cause: str, task_id: str | None
    ) -> MinerBudget:
        miner = self.get_or_create(hotkey, pool)
        new = max(1, min(CEILING, new))
        self.db.execute(
            "UPDATE miners SET budget = ? WHERE hotkey = ? AND pool = ?",
            (new, hotkey, pool),
        )
        self.db.execute(
            "INSERT INTO budget_events"
            " (hotkey, pool, old_budget, new_budget, cause, task_id, at)"
            " VALUES (?, ?, ?, ?, ?, ?, ?)",
            (hotkey, pool, miner.budget, new, cause, task_id, time.time()),
        )
        self.db.commit()
        return MinerBudget(hotkey, pool, new, miner.verified)

    def reward(
        self,
        hotkey: str,
        task_id: str,
        amount: int,
        ramp: bool = True,
        pool: str = CRAWL,
    ) -> MinerBudget:
        miner = self.get_or_create(hotkey, pool)
        self.db.execute(
            "UPDATE miners SET verified = verified + ? WHERE hotkey = ? AND pool = ?",
            (amount, hotkey, pool),
        )
        self.credit(hotkey, amount, pool)
        if not ramp:
            self.db.commit()
            return self.get_or_create(hotkey, pool)
        grown = miner.budget + max(1, int(miner.budget * GROWTH))
        return self._set_budget(hotkey, pool, grown, REWARD, task_id)

    def credit(self, hotkey: str, amount: int, pool: str = CRAWL) -> None:
        """Rows toward the miner's share this hour; a bad task takes its URLs back."""
        self.db.execute(
            "INSERT INTO credits (pool, hotkey, hour, amount) VALUES (?, ?, ?, ?)"
            " ON CONFLICT (pool, hotkey, hour) DO UPDATE SET amount = amount + excluded.amount",
            (pool, hotkey, hour_of(), amount),
        )
        self.db.commit()

    def wipe_credits(self, hotkey: str, since: float, pool: str = CRAWL) -> None:
        self.db.execute(
            "DELETE FROM credits WHERE hotkey = ? AND pool = ? AND hour >= ?",
            (hotkey, pool, hour_of(since)),
        )
        self.db.commit()

    def penalise(
        self, hotkey: str, task_id: str, cause: str, pool: str = CRAWL
    ) -> MinerBudget:
        assert cause in PENALTIES, cause
        halved = self.get_or_create(hotkey, pool).budget // 2
        return self._set_budget(hotkey, pool, halved, cause, task_id)

    def strike(
        self,
        hotkey: str,
        reason: str,
        task_id: str,
        judged: int,
        pool: str = CRAWL,
        now: float | None = None,
    ) -> float | None:
        """The lockout's end if this strike, among `judged` recent verdicts, starts one."""
        now = now or time.time()
        if reason in EXPIRY_REASONS and self._lapsed_recently(hotkey, pool, now):
            return self.locked_until(hotkey, pool, now)
        self.db.execute(
            "INSERT INTO strikes (hotkey, pool, reason, task_id, at)"
            " VALUES (?, ?, ?, ?, ?)",
            (hotkey, pool, reason, task_id, now),
        )
        (recent,) = self.db.execute(
            "SELECT COUNT(*) FROM strikes WHERE hotkey = ? AND pool = ? AND at > ?",
            (hotkey, pool, now - STRIKE_WINDOW_H * HOUR),
        ).fetchone()
        until = None
        if recent >= STRIKES_TO_LOCK and recent >= STRIKE_SHARE * judged:
            (before,) = self.db.execute(
                "SELECT COUNT(*) FROM lockout_history WHERE hotkey = ? AND pool = ?"
                " AND at > ?",
                (hotkey, pool, now - LOCKOUT_MEMORY_H * HOUR),
            ).fetchone()
            hours = LOCKOUT_STEPS_H[min(before, len(LOCKOUT_STEPS_H) - 1)]
            until = self.lock_out(hotkey, pool, hours, reason, now)
        self.db.commit()
        return until

    def _lapsed_recently(self, hotkey: str, pool: str, now: float) -> bool:
        placeholders = ", ".join("?" * len(EXPIRY_REASONS))
        return bool(
            self.db.execute(
                "SELECT 1 FROM strikes WHERE hotkey = ? AND pool = ? AND at > ?"
                f" AND reason IN ({placeholders}) LIMIT 1",
                (hotkey, pool, now - STRIKE_BURST_S, *EXPIRY_REASONS),
            ).fetchone()
        )

    def lock_out(
        self,
        hotkey: str,
        pool: str,
        hours: float,
        reason: str,
        now: float | None = None,
    ) -> float:
        now = now or time.time()
        until = now + hours * HOUR
        self.db.execute(
            "INSERT INTO lockouts (hotkey, pool, until, reason) VALUES (?, ?, ?, ?)"
            " ON CONFLICT (hotkey, pool) DO UPDATE SET until = MAX(until, excluded.until),"
            " reason = excluded.reason",
            (hotkey, pool, until, reason),
        )
        self.db.execute(
            "INSERT INTO lockout_history (hotkey, pool, at) VALUES (?, ?, ?)",
            (hotkey, pool, now),
        )
        self.db.commit()
        return self.locked_until(hotkey, pool, now) or until

    def locked_until(
        self, hotkey: str, pool: str = CRAWL, now: float | None = None
    ) -> float | None:
        row = self.db.execute(
            "SELECT until FROM lockouts WHERE hotkey = ? AND pool = ?", (hotkey, pool)
        ).fetchone()
        return row[0] if row and row[0] > (now or time.time()) else None

    def all(self, pool: str = CRAWL) -> list[MinerBudget]:
        rows = self.db.execute(
            "SELECT hotkey, pool, budget, verified FROM miners WHERE pool = ?", (pool,)
        ).fetchall()
        return [MinerBudget(*row) for row in rows]

    def lockouts(self, pool: str = CRAWL, now: float | None = None) -> dict[str, float]:
        return dict(
            self.db.execute(
                "SELECT hotkey, until FROM lockouts WHERE pool = ? AND until > ?",
                (pool, now or time.time()),
            ).fetchall()
        )

    def pools_of(self, hotkey: str) -> list[MinerBudget]:
        rows = self.db.execute(
            "SELECT hotkey, pool, budget, verified FROM miners WHERE hotkey = ?"
            " ORDER BY pool",
            (hotkey,),
        ).fetchall()
        return [MinerBudget(*row) for row in rows]

    def history(self, hotkey: str, limit: int = 100) -> list[dict]:
        rows = self.db.execute(
            "SELECT pool, old_budget, new_budget, cause, task_id, at FROM budget_events"
            " WHERE hotkey = ? ORDER BY id DESC LIMIT ?",
            (hotkey, limit),
        ).fetchall()
        return [
            {
                "pool": r[0],
                "old": r[1],
                "new": r[2],
                "cause": r[3],
                "task_id": r[4],
                "at": r[5],
            }
            for r in rows
        ]

    def shares(
        self, window_hours: int = SHARE_WINDOW_H, now: float | None = None
    ) -> dict[str, dict[str, float]]:
        since = hour_of(now) - window_hours + 1
        earned: dict[str, dict[str, int]] = {}
        for pool, hotkey, amount in self.db.execute(
            "SELECT pool, hotkey, SUM(amount) FROM credits"
            " WHERE hour >= ? AND hour <= ? GROUP BY pool, hotkey",
            (since, hour_of(now)),
        ):
            if amount <= 0:
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
            " WHERE hour >= ? AND hour <= ? GROUP BY hotkey HAVING SUM(assigned) > 0",
            (since, hour_of(now)),
        ).fetchall()
        return {
            hotkey: {
                "assigned": assigned,
                "returned": returned,
                "coverage": round(returned / assigned, 4),
            }
            for hotkey, assigned, returned in rows
        }

    def prune(self, keep_hours: int = KEEP_H) -> None:
        oldest = hour_of() - keep_hours
        self.db.execute("DELETE FROM credits WHERE hour < ?", (oldest,))
        self.db.execute("DELETE FROM coverage WHERE hour < ?", (oldest,))
        self.db.execute("DELETE FROM strikes WHERE at < ?", (oldest * HOUR,))
        self.db.execute(
            "DELETE FROM lockout_history WHERE at < ?",
            (time.time() - LOCKOUT_MEMORY_H * HOUR,),
        )
        self.db.commit()
