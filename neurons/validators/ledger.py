"""The verdicts this validator reached itself and every miner's logged uploads, from which it sets its own weights."""

from __future__ import annotations

import json
import sqlite3
import time
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from desearch.credit import (
    COVERAGE_GATE,
    FAILS_FOR_PENALTY,
    PENALTY_WINDOW_S,
    RATE_WINDOW_S,
    RECENT_CHECKS,
    SHARE_WINDOW_H,
    covers,
    neutral_hosts,
    outside_hosts,
    outside_neutral,
    overstated,
    round_gate,
)
from desearch.kinds import CRAWL, EMBED

# A check whose upload storage served in another version than the one pinned: evidence against storage, not against the
# miner, so it neither pays nor costs, and the upload is not paid as an unchecked one either.
NEUTRAL = "neutral"


# A failed crawl task takes its assigned URLs back, so failing costs more than not trying.
NET = f"SUM(CASE WHEN kind = '{CRAWL}' AND verdict = 'fail' THEN -assigned ELSE credited END)"
CRAWL_RETURNED = "MIN(ok + errors, assigned)"
EMBED_RETURNED = "chars"
# Ledgers written before embed uploads were logged hold only crawl rows.
ADDED_COLUMNS = (
    ("uploads", "kind", f"TEXT NOT NULL DEFAULT '{CRAWL}'"),
    ("uploads", "chars", "INTEGER NOT NULL DEFAULT 0"),
    ("checks", "kind", f"TEXT NOT NULL DEFAULT '{CRAWL}'"),
    # The round and rows per host, so hosts most miners of a round could not load pay nobody.
    ("uploads", "round_id", "TEXT NOT NULL DEFAULT ''"),
    ("checks", "round_id", "TEXT NOT NULL DEFAULT ''"),
    ("checks", "hosts", "TEXT"),
    # Coverage is judged when weights are set, against the round's gate then: the verdict without it, the task's URLs per
    # host, and the samples this validator could load. Older rows have none and keep the verdict they were recorded with.
    ("checks", "content_verdict", "TEXT"),
    ("checks", "assigned_hosts", "TEXT"),
    ("checks", "sampled", "INTEGER NOT NULL DEFAULT 0"),
    ("checks", "loaded", "INTEGER NOT NULL DEFAULT 0"),
)


class Ledger:
    def __init__(self, path: Path | str):
        self.db = sqlite3.connect(str(path), check_same_thread=False)
        self.db.executescript("""
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
            """)
        for table, column, declared in ADDED_COLUMNS:
            have = {row[1] for row in self.db.execute(f"PRAGMA table_info({table})")}
            if column not in have:
                self.db.execute(f"ALTER TABLE {table} ADD COLUMN {column} {declared}")
        self.db.execute(
            "CREATE INDEX IF NOT EXISTS checks_round ON checks (round_id, kind)"
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
            "INSERT OR IGNORE INTO uploads (key, task_id, miner, completed_at, assigned,"
            " ok, errors, kind, chars, round_id) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [upload_row(e) for e in entries],
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
        kind: str = CRAWL,
        round_id: str = "",
        hosts: dict[str, list[int]] | None = None,
        content_verdict: str | None = None,
        assigned_hosts: dict[str, int] | None = None,
        sampled: int = 0,
        loaded: int = 0,
    ) -> None:
        """A check of one upload. For a crawl check, `verdict` and `credited` are its verdict without coverage and what it
        pays if coverage holds, given as `content_verdict`; coverage is judged when weights are set.
        """
        self.db.execute(
            "INSERT OR REPLACE INTO checks (key, miner, completed_at, scored_at, verdict,"
            " credited, returned, content, assigned, kind, round_id, hosts,"
            " content_verdict, assigned_hosts, sampled, loaded)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
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
                kind,
                round_id,
                json.dumps(hosts) if hosts is not None else None,
                content_verdict,
                json.dumps(assigned_hosts) if assigned_hosts is not None else None,
                sampled,
                loaded,
            ),
        )
        self.db.commit()

    def round_view(
        self, round_id: str, settled: Mapping | None = None
    ) -> tuple[float, set[str]]:
        """A round's coverage gate and neutral hosts: as the task API settled them in its signed log once the round closed,
        else as this validator's own checks of the round show them so far."""
        if settled:
            return view_of(settled)
        if not round_id:
            return COVERAGE_GATE, set()
        found = self.db.execute(
            "SELECT hosts, sampled, loaded FROM checks WHERE round_id = ? AND kind = ?",
            (round_id, CRAWL),
        ).fetchall()
        checked = [json.loads(hosts) for hosts, _, _ in found if hosts]
        sampled = sum(row[1] for row in found)
        loaded = sum(row[2] for row in found)
        return round_gate(sampled, loaded), neutral_hosts(checked)

    def state(self, name: str) -> str | None:
        found = self.db.execute(
            "SELECT value FROM state WHERE name = ?", (name,)
        ).fetchone()
        return found[0] if found else None

    def set_state(self, name: str, value: str) -> None:
        self.db.execute("INSERT OR REPLACE INTO state VALUES (?, ?)", (name, value))
        self.db.commit()

    def crawl_paid(
        self,
        now: float | None = None,
        rounds: Mapping[str, Mapping] | None = None,
        own_only: bool = False,
        excluded: list[tuple[float, float]] = (),
    ) -> dict[str, Paid]:
        """Each miner's rows of the scoring window as this validator's own checks judge them, each round's coverage gate and
        neutral hosts as the task API's signed log settled them (`rounds`, by round id), else as our own checks show them.
        With `own_only`, the uploads the task API logged count for nothing, only those this validator checked; uploads logged
        within `excluded` never count.
        """
        return self._paid(CRAWL, CRAWL_RETURNED, now, rounds, own_only, excluded)

    def embed_paid(
        self,
        now: float | None = None,
        own_only: bool = False,
        excluded: list[tuple[float, float]] = (),
    ) -> dict[str, Paid]:
        """Each miner's characters of the scoring window as this validator's own checks judge them; `own_only` and `excluded` as
        for `crawl_paid`."""
        return self._paid(EMBED, EMBED_RETURNED, now, None, own_only, excluded)

    def _paid(
        self,
        kind: str,
        returned_of: str,
        now: float | None,
        settled: Mapping[str, Mapping] | None = None,
        own_only: bool = False,
        excluded: list[tuple[float, float]] = (),
    ) -> dict[str, Paid]:
        now = now or time.time()
        since = now - SHARE_WINDOW_H * 3600
        found = self.db.execute(
            "SELECT c.miner, c.key, c.completed_at, c.verdict, c.credited, c.returned,"
            " c.content, c.assigned, u.ok, c.round_id, c.hosts, c.content_verdict,"
            " c.assigned_hosts, c.sampled, c.loaded FROM checks c"
            " LEFT JOIN uploads u ON u.key = c.key"
            " WHERE c.scored_at >= ? AND c.kind = ?"
            " ORDER BY c.scored_at",
            (now - RATE_WINDOW_S, kind),
        ).fetchall()
        rounds = round_views(found, settled or {})
        checks: dict[str, list[Check]] = defaultdict(list)
        unpaid: set[str] = set()
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
            round_id,
            hosts,
            content_verdict,
            assigned_hosts,
            _,
            _,
        ) in found:
            if verdict == NEUTRAL:
                unpaid.add(key)
                continue
            gate, neutral, _ = rounds.get(round_id, (COVERAGE_GATE, set(), 0.0))
            per_host = json.loads(hosts) if hosts else {}
            urls = json.loads(assigned_hosts) if assigned_hosts else None
            if content_verdict is None or urls is None:
                # Recorded with coverage already judged, or an embed check.
                final = verdict
                kept_urls = None
            else:
                rows_out, kept_urls = outside_hosts(per_host, urls, set(neutral))
                final = judged(content_verdict, covers(rows_out, kept_urls, gate))
            if final not in ("pass", "fail"):
                continue
            passed = final == "pass" and not (
                kind == CRAWL and ok is not None and overstated(ok, content, assigned)
            )
            # Rows on hosts most miners of the round could not load pay nobody and cost nobody.
            kept = outside_neutral(per_host, set(neutral))
            checks[miner].append(
                Check(
                    key,
                    completed_at,
                    passed,
                    round(credited * kept) if passed else 0,
                    round(returned * kept),
                    kept_urls if kept_urls is not None else round(assigned * kept),
                )
            )
        uploads: dict[str, list[tuple[str, float, int]]] = defaultdict(list)
        for miner, key, completed_at, returned, round_id in self.db.execute(
            f"SELECT miner, key, completed_at, {returned_of}, round_id FROM uploads"
            " WHERE completed_at >= ? AND kind = ?",
            (since, kind),
        ):
            if key in unpaid:
                continue
            # The task API's log is all that says these uploads happened; while it cannot be trusted, they count for nothing.
            if own_only or any(start <= completed_at <= end for start, end in excluded):
                continue
            neutral_share = rounds.get(round_id, (COVERAGE_GATE, set(), 0.0))[2]
            uploads[miner].append((key, completed_at, returned * (1.0 - neutral_share)))
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
    amount: float
    uploads: int
    checked: int
    failed: int
    rate: float


def view_of(settled: Mapping) -> tuple[float, set[str]]:
    """The gate and neutral hosts of a round as the task API's `close` entry settled them."""
    return float(settled.get("gate", COVERAGE_GATE)), set(settled.get("neutral") or ())


def judged(content_verdict: str, covered: bool) -> str:
    """A crawl check's verdict once coverage is known: its own fail stands, too few rows fail it, and otherwise it is what
    the pages showed (a void stays void)."""
    if content_verdict == "fail" or not covered:
        return "fail"
    return content_verdict


def round_views(
    found: list[tuple], settled: Mapping[str, Mapping]
) -> dict[str, tuple[float, set[str], float]]:
    """Per round this validator checked: its coverage gate and neutral hosts, from the task API's settled view when the
    round has closed, else from our own checks (their loadability and the hosts most of them could not load); and the
    share of our checked rows on its neutral hosts, every upload's share too since every task of a round has its mix.
    """
    by_round: dict[str, list[dict[str, list[int]]]] = defaultdict(list)
    loads: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for row in found:
        round_id, hosts, sampled, loaded = row[9], row[10], row[13], row[14]
        if not round_id:
            continue
        loads[round_id][0] += sampled or 0
        loads[round_id][1] += loaded or 0
        if hosts:
            by_round[round_id].append(json.loads(hosts))
    rounds = {}
    for round_id in loads:
        checked = by_round.get(round_id, [])
        if settled.get(round_id):
            gate, neutral = view_of(settled[round_id])
        else:
            gate, neutral = round_gate(*loads[round_id]), neutral_hosts(checked)
        total = sum(rows for hosts in checked for rows, _ in hosts.values())
        on_neutral = sum(
            rows
            for hosts in checked
            for host, (rows, _) in hosts.items()
            if host in neutral
        )
        rounds[round_id] = (gate, neutral, on_neutral / total if total else 0.0)
    return rounds


def upload_row(entry: dict) -> tuple:
    """A logged upload as stored; an entry without a kind is a crawl upload from an older log."""
    if entry.get("kind", CRAWL) == EMBED:
        counts = (entry["texts"], entry["texts"], 0, EMBED, entry["chars"])
    else:
        counts = (entry["assigned"], entry["ok"], entry["errors"], CRAWL, 0)
    return (
        entry["key"],
        entry["task_id"],
        entry["miner"],
        entry["completed_at"],
        *counts,
        entry.get("round_id") or "",
    )


def paid(
    checks: list[Check], uploads: list[tuple[str, float, int]], since: float
) -> Paid:
    """Checked uploads at what they paid, the others at the rate the checks paid, none of what a failed check takes back."""
    judged = sum(check.returned for check in checks)
    rate = sum(check.credited for check in checks) / judged if judged else 0.0
    taken = taken_back(checks)
    checked = {check.key for check in checks}
    recent = [check for check in checks if check.completed_at >= since]
    amount = 0.0
    for check in recent:
        if not check.passed:
            amount -= check.assigned
        elif not taken_at(taken, check.completed_at):
            amount += check.credited
    for key, completed_at, returned in uploads:
        if key not in checked and not taken_at(taken, completed_at):
            amount += rate * returned
    failed = sum(1 for check in recent if not check.passed)
    return Paid(amount, len(uploads), len(recent), failed, round(rate, 4))


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
