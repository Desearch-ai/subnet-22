from __future__ import annotations

import json
import sqlite3
import time
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone

from desearch.credit import EVIDENCE, credited_urls

from .budget import COVERAGE_GATE

__all__ = ["credited_urls"]

COUNTS = (
    "returned",
    "missing",
    "duplicates",
    "sampled",
    "matched",
    "mismatched",
    "unverifiable",
    "errors_confirmed",
    "errors_unconfirmed",
    "reextract_mismatch",
    "credited",
)
FIELDS = (
    "task_id",
    "kind",
    "round_id",
    "miner",
    "validator",
    "verdict",
    "reason",
    *COUNTS,
    "upload_key",
    "page_key",
    "report_key",
    "claimed_at",
    "completed_at",
    "scored_at",
)
VOTE_COUNTS = (
    "returned",
    "sampled",
    "matched",
    "mismatched",
    "unverifiable",
    "errors_confirmed",
    "errors_unconfirmed",
    "credited",
)
VOTE_FIELDS = (
    "task_id",
    "kind",
    "miner",
    "validator",
    "verdict",
    "reason",
    *VOTE_COUNTS,
    "final_verdict",
    "final_credited",
    "agreed",
    "decided",
    "voted_at",
    "finalized_at",
)
VOTE_COLUMNS = (*VOTE_FIELDS[:4], "upload_key", *VOTE_FIELDS[4:])
VERDICTS = ("pass", "fail", "void")
NO_MAJORITY = "validators_disagree"
WITHDRAWN = "withdrawn"
MIN_AUDITS = 10
URL_DETAIL_DAYS = 7
MAX_DISAGREEMENT = 0.3
MAX_CREDIT_DIVERGENCE = 0.15
# The validator's own timeout or crash, never held against the miner.
VALIDATOR_FAULT_REASONS = frozenset({"unscorable"})
# A page unreachable both for the miner and the validator (not_fetched) can sit on either kind of row.
CONTENT_OUTCOMES = ("matched", "mismatched", "unverifiable")
ERROR_OUTCOMES = ("errors_confirmed", "errors_unconfirmed")


class Infeasible(ValueError):
    pass


@dataclass
class Decision:
    outcome: str
    vote: dict | None = None
    votes: list[dict] = field(default_factory=list)
    agreed: list[str] = field(default_factory=list)
    disagreed: list[str] = field(default_factory=list)


def check_feasible(job: dict, result: dict) -> None:
    """A report must describe rows the task can hold; counts are not taken on trust."""
    urls = set(job["urls"])
    samples = result.get("samples", [])
    sampled = [sample["url"] for sample in samples]
    outcomes = Counter(sample["outcome"] for sample in samples)
    returned, error_rows = result["returned"], result.get("error_rows", 0)
    content = sum(outcomes[outcome] for outcome in CONTENT_OUTCOMES)
    errors = sum(outcomes[outcome] for outcome in ERROR_OUTCOMES)
    problems = (
        (len(set(sampled)) < len(sampled), "a URL was sampled twice"),
        (not set(sampled) <= urls, "a sampled URL is not in the task"),
        (returned > len(urls), "more rows returned than URLs assigned"),
        (len(sampled) > returned, "more samples than rows returned"),
        (content > returned - error_rows, "more content samples than content rows"),
        (errors > error_rows, "more error samples than error rows"),
        (
            result["verdict"] == "pass" and returned < COVERAGE_GATE * len(urls),
            f"a pass needs {COVERAGE_GATE:.0%} of the URLs returned",
        ),
    )
    for wrong, why in problems:
        if wrong:
            raise Infeasible(why)


def build_vote(job: dict, validator: str, result: dict) -> dict:
    """Credit comes from the samples, never from the validator's own number."""
    check_feasible(job, result)
    returned = result["returned"]
    error_rows = result.get("error_rows", 0)
    samples = result.get("samples", [])
    outcomes = Counter(sample["outcome"] for sample in samples)
    verdict, reason = result["verdict"], result.get("reason", "")
    if verdict == "fail" and reason in VALIDATOR_FAULT_REASONS:
        verdict = "void"
    elif verdict == "pass" and not any(outcomes[outcome] for outcome in EVIDENCE):
        verdict, reason = "void", "inconclusive"
    elif verdict == "pass" and outcomes["unverifiable"] * 2 >= len(samples):
        verdict, reason = "void", "unverifiable"
    credited = (
        credited_urls(returned - error_rows, error_rows, outcomes)
        if verdict == "pass"
        else 0
    )
    return {
        "validator": validator,
        "verdict": verdict,
        "credited": credited,
        "result": {
            **result,
            "verdict": verdict,
            "reason": reason,
            "returned": returned,
            "error_rows": error_rows,
            "credited": credited,
            "rejected": sorted(set(result.get("rejected", [])) & set(job["urls"])),
        },
    }


def check_embed_feasible(job: dict, result: dict) -> None:
    """A pass must account for every text; a validator's counts are not taken on trust."""
    samples = result.get("samples", [])
    ids = [sample["text_id"] for sample in samples]
    outcomes = Counter(sample["outcome"] for sample in samples)
    texts = job.get("texts", 0)
    problems = (
        (len(set(ids)) < len(ids), "a text was sampled twice"),
        (result["returned"] > texts, "more vectors returned than texts assigned"),
        (len(ids) > result["returned"], "more samples than vectors returned"),
        (
            result["verdict"] == "pass"
            and (
                result["returned"] < texts
                or result.get("missing")
                or result.get("duplicates")
                or result.get("malformed")
                or outcomes["mismatched"]
            ),
            "a pass needs every text embedded and every sample matched",
        ),
    )
    for wrong, why in problems:
        if wrong:
            raise Infeasible(why)


def build_embed_vote(job: dict, validator: str, result: dict) -> dict:
    """Credit is the characters the API assigned, paid in full on a pass."""
    check_embed_feasible(job, result)
    verdict, reason = result["verdict"], result.get("reason", "")
    if verdict == "pass" and not result.get("matched"):
        verdict, reason = "void", "inconclusive"
    credited = job.get("chars", 0) if verdict == "pass" else 0
    return {
        "validator": validator,
        "verdict": verdict,
        "credited": credited,
        "result": {
            **result,
            "verdict": verdict,
            "reason": reason,
            "credited": credited,
        },
    }


def decide(votes: list[dict], audit: bool = False, overdue: bool = False) -> Decision:
    """Audited votes need two to agree; a third breaks a tie."""
    if len(votes) == 1:
        return (
            Decision("audit")
            if audit and not overdue
            else Decision("final", votes[0], votes)
        )

    verdict, count = Counter(v["verdict"] for v in votes).most_common(1)[0]
    if count * 2 <= len(votes):
        if len(votes) < 3 and not overdue:
            return Decision("audit")
        standing = {**votes[-1], "verdict": "void", "credited": 0}
        standing["result"] = {
            **standing["result"],
            "verdict": "void",
            "reason": NO_MAJORITY,
        }
        return Decision("final", standing, votes)

    winners = [v for v in votes if v["verdict"] == verdict]
    losers = [v for v in votes if v["verdict"] != verdict]
    if credits_diverge(winners):
        if len(votes) < 3:
            if not overdue:
                return Decision("audit")
            return Decision("final", min(winners, key=credited), votes)
        # The lower median, so two passes that differ finalize on the cheaper one.
        middle = sorted(v["credited"] for v in winners)[(len(winners) - 1) // 2]
        near = [v for v in winners if credits_agree(v["credited"], middle)]
        losers += [v for v in winners if v not in near]
        winners = near
    return Decision(
        "final",
        min(winners, key=credited),
        votes,
        agreed=[v["validator"] for v in winners],
        disagreed=[v["validator"] for v in losers],
    )


def credited(vote: dict) -> int:
    return vote["credited"]


def credits_agree(one: int, other: int) -> bool:
    return abs(one - other) <= MAX_CREDIT_DIVERGENCE * max(one, other)


def credits_diverge(votes: list[dict]) -> bool:
    paid = [v["credited"] for v in votes]
    return len(paid) > 1 and not credits_agree(min(paid), max(paid))


class Validations:
    def __init__(self, db: sqlite3.Connection):
        self.db = db
        counts = "".join(f"{name} INTEGER NOT NULL DEFAULT 0, " for name in COUNTS)
        vote_counts = "".join(
            f"{name} INTEGER NOT NULL DEFAULT 0, " for name in VOTE_COUNTS
        )
        self.db.executescript(
            f"""
            CREATE TABLE IF NOT EXISTS validations (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                task_id    TEXT NOT NULL,
                kind       TEXT NOT NULL,
                round_id   TEXT NOT NULL,
                miner      TEXT NOT NULL,
                validator  TEXT NOT NULL,
                verdict    TEXT NOT NULL,
                reason     TEXT NOT NULL,
                {counts}
                upload_key TEXT NOT NULL,
                page_key   TEXT,
                report_key TEXT NOT NULL,
                claimed_at REAL,
                completed_at REAL,
                scored_at  REAL NOT NULL,
                report     TEXT NOT NULL,
                urls       TEXT
            );
            CREATE INDEX IF NOT EXISTS validations_task ON validations (task_id, id);
            CREATE INDEX IF NOT EXISTS validations_miner ON validations (miner, verdict);
            CREATE INDEX IF NOT EXISTS validations_scored ON validations (scored_at);
            CREATE INDEX IF NOT EXISTS validations_miner_scored ON validations (miner, scored_at);
            CREATE INDEX IF NOT EXISTS validations_validator_scored ON validations (validator, scored_at);
            CREATE TABLE IF NOT EXISTS verdict_counts (
                miner   TEXT NOT NULL,
                verdict TEXT NOT NULL,
                n       INTEGER NOT NULL,
                PRIMARY KEY (miner, verdict)
            );
            CREATE TABLE IF NOT EXISTS validator_audits (
                hotkey        TEXT PRIMARY KEY,
                audits        INTEGER NOT NULL DEFAULT 0,
                disagreements INTEGER NOT NULL DEFAULT 0
            );
            CREATE TABLE IF NOT EXISTS final_verdicts (
                task_id    TEXT NOT NULL,
                upload_key TEXT NOT NULL,
                verdict    TEXT NOT NULL,
                credited   INTEGER NOT NULL,
                publish    TEXT,
                finalized_at REAL NOT NULL,
                PRIMARY KEY (task_id, upload_key)
            );
            CREATE TABLE IF NOT EXISTS retention (name TEXT PRIMARY KEY, mark REAL NOT NULL);
            CREATE TABLE IF NOT EXISTS votes (
                id             INTEGER PRIMARY KEY AUTOINCREMENT,
                task_id        TEXT NOT NULL,
                kind           TEXT NOT NULL,
                miner          TEXT NOT NULL,
                validator      TEXT NOT NULL,
                upload_key     TEXT NOT NULL,
                verdict        TEXT NOT NULL,
                reason         TEXT NOT NULL,
                {vote_counts}
                final_verdict  TEXT NOT NULL,
                final_credited INTEGER NOT NULL,
                agreed         INTEGER,
                decided        INTEGER NOT NULL,
                voted_at       REAL NOT NULL,
                finalized_at   REAL NOT NULL
            );
            CREATE INDEX IF NOT EXISTS votes_task ON votes (task_id, id);
            CREATE INDEX IF NOT EXISTS votes_validator ON votes (validator, id);
            CREATE INDEX IF NOT EXISTS votes_miner ON votes (miner, id);
            CREATE INDEX IF NOT EXISTS votes_finalized ON votes (finalized_at);
            """
        )
        if not self.db.execute("SELECT 1 FROM verdict_counts LIMIT 1").fetchone():
            self.db.execute(
                "INSERT INTO verdict_counts"
                " SELECT miner, verdict, COUNT(*) FROM validations GROUP BY miner, verdict"
            )
        self.db.commit()

    def finalize(
        self,
        task_id: str,
        upload_key: str,
        verdict: str,
        credited: int,
        publish: dict | None,
    ) -> bool:
        """False when this upload was finalized before, so nothing is paid twice."""
        try:
            self.db.execute(
                "INSERT INTO final_verdicts VALUES (?, ?, ?, ?, ?, ?)",
                (
                    task_id,
                    upload_key,
                    verdict,
                    credited,
                    json.dumps(publish) if publish else None,
                    time.time(),
                ),
            )
        except sqlite3.IntegrityError:
            return False
        self.db.commit()
        return True

    def final_verdict(self, task_id: str, upload_key: str) -> dict | None:
        row = self.db.execute(
            "SELECT verdict, credited, publish FROM final_verdicts"
            " WHERE task_id = ? AND upload_key = ?",
            (task_id, upload_key),
        ).fetchone()
        if row is None:
            return None
        verdict, credited, publish = row
        return {
            "verdict": verdict,
            "credited": credited,
            "publish": json.loads(publish) if publish else None,
        }

    def record(self, report: dict, urls: list[dict] | None = None) -> None:
        columns = (*FIELDS, "report", "urls")
        self.db.execute(
            f"INSERT INTO validations ({', '.join(columns)})"
            f" VALUES ({', '.join('?' * len(columns))})",
            (
                *(report[name] for name in FIELDS),
                json.dumps(report, sort_keys=True),
                json.dumps(urls) if urls else None,
            ),
        )
        self.db.execute(
            "INSERT INTO verdict_counts VALUES (?, ?, 1)"
            " ON CONFLICT (miner, verdict) DO UPDATE SET n = n + 1",
            (report["miner"], report["verdict"]),
        )
        self.db.commit()

    def record_votes(
        self,
        report: dict,
        votes: list[dict],
        disagreed: tuple[str, ...] | list[str] = (),
        decided: bool = True,
    ) -> None:
        """Every validator's vote on one upload; `decided` is false when no vote carried it."""
        for vote in votes:
            result, validator = vote["result"], vote["validator"]
            self.db.execute(
                f"INSERT INTO votes ({', '.join(VOTE_COLUMNS)})"
                f" VALUES ({', '.join('?' * len(VOTE_COLUMNS))})",
                (
                    report["task_id"],
                    report["kind"],
                    report["miner"],
                    validator,
                    report["upload_key"],
                    vote["verdict"],
                    result.get("reason", ""),
                    *(result.get(name, 0) for name in VOTE_COUNTS),
                    report["verdict"],
                    report["credited"],
                    validator not in disagreed if decided else None,
                    decided and validator == report["validator"],
                    vote.get("at", report["scored_at"]),
                    report["scored_at"],
                ),
            )
        self.db.commit()

    def votes(
        self,
        validator: str | None = None,
        miner: str | None = None,
        task_id: str | None = None,
        verdict: str | None = None,
        agreed: bool | None = None,
        until: float | None = None,
        before: int | None = None,
        limit: int = 50,
    ) -> tuple[list[dict], int | None]:
        """Votes newest first, and the cursor of the next page when there is one."""
        where, args = ["1"], []
        for column, value in (
            ("validator", validator),
            ("miner", miner),
            ("task_id", task_id),
            ("verdict", verdict),
            ("agreed", agreed),
        ):
            if value is not None:
                where.append(f"{column} = ?")
                args.append(value)
        if until is not None:
            where.append("finalized_at <= ?")
            args.append(until)
        if before is not None:
            where.append("id < ?")
            args.append(before)
        rows = self.db.execute(
            f"SELECT id, {', '.join(VOTE_FIELDS)} FROM votes"
            f" WHERE {' AND '.join(where)} ORDER BY id DESC LIMIT ?",
            (*args, limit),
        ).fetchall()
        return (
            [as_vote(row[1:]) for row in rows],
            rows[-1][0] if len(rows) == limit else None,
        )

    def votes_on(self, task_id: str, upload_key: str) -> list[dict]:
        rows = self.db.execute(
            f"SELECT {', '.join(VOTE_FIELDS)} FROM votes"
            " WHERE task_id = ? AND upload_key = ? ORDER BY id",
            (task_id, upload_key),
        ).fetchall()
        return [as_vote(row) for row in rows]

    def validator_totals(self, since: float, until: float) -> dict[str, dict]:
        rows = self.db.execute(
            f"SELECT validator, COUNT(*), {VERDICT_SUMS}, SUM(agreed = 1), SUM(agreed = 0),"
            " SUM(decided), MAX(voted_at) FROM votes"
            " WHERE finalized_at >= ? AND finalized_at <= ? GROUP BY validator",
            (since, until),
        ).fetchall()
        return {
            validator: {
                "votes": votes,
                "pass": passed,
                "fail": failed,
                "void": void,
                "agreed": agreed or 0,
                "disagreed": disagreed or 0,
                "decided": decided,
                "last_vote_at": last,
            }
            for validator, votes, passed, failed, void, agreed, disagreed, decided, last in rows
        }

    def miner_totals(self, since: float, until: float) -> dict[str, dict]:
        rows = self.db.execute(
            f"SELECT miner, COUNT(*), {VERDICT_SUMS}, SUM(returned), SUM(missing),"
            " SUM(credited), MAX(scored_at) FROM validations"
            " WHERE scored_at >= ? AND scored_at <= ? GROUP BY miner",
            (since, until),
        ).fetchall()
        return {
            miner: {
                "tasks": tasks,
                "pass": passed,
                "fail": failed,
                "void": void,
                "returned": returned,
                "missing": missing,
                "credited": credited,
                "last_scored_at": last,
            }
            for miner, tasks, passed, failed, void, returned, missing, credited, last in rows
        }

    def task_series(
        self, bucket_s: int, since: float, until: float, miner: str | None = None
    ) -> dict[int, dict]:
        rows = self.db.execute(
            f"SELECT CAST(scored_at / ? AS INTEGER), COUNT(*), {VERDICT_SUMS},"
            " SUM(returned), SUM(credited) FROM validations"
            " WHERE scored_at >= ? AND scored_at <= ?"
            + (" AND miner = ?" if miner else "")
            + " GROUP BY 1",
            (bucket_s, since, until, *([miner] if miner else [])),
        ).fetchall()
        names = ("tasks", *VERDICTS, "returned", "credited")
        return {row[0]: dict(zip(names, row[1:], strict=True)) for row in rows}

    def vote_series(
        self, bucket_s: int, since: float, until: float, validator: str
    ) -> dict[int, dict]:
        rows = self.db.execute(
            f"SELECT CAST(finalized_at / ? AS INTEGER), COUNT(*), {VERDICT_SUMS},"
            " COALESCE(SUM(agreed = 1), 0), COALESCE(SUM(agreed = 0), 0) FROM votes"
            " WHERE finalized_at >= ? AND finalized_at <= ? AND validator = ? GROUP BY 1",
            (bucket_s, since, until, validator),
        ).fetchall()
        names = ("votes", *VERDICTS, "agreed", "disagreed")
        return {row[0]: dict(zip(names, row[1:], strict=True)) for row in rows}

    def latest(self, task_id: str) -> dict | None:
        row = self.db.execute(
            f"SELECT {', '.join(FIELDS)} FROM validations"
            " WHERE task_id = ? ORDER BY id DESC LIMIT 1",
            (task_id,),
        ).fetchone()
        return dict(zip(FIELDS, row, strict=True)) if row else None

    def uploads(self, task_id: str) -> list[dict]:
        """Every finalized upload of a task, newest first."""
        rows = self.db.execute(
            f"SELECT {', '.join(FIELDS)} FROM validations"
            " WHERE task_id = ? ORDER BY id DESC",
            (task_id,),
        ).fetchall()
        return [dict(zip(FIELDS, row, strict=True)) for row in rows]

    def urls(self, task_id: str) -> list[dict]:
        row = self.db.execute(
            "SELECT urls FROM validations WHERE task_id = ? ORDER BY id DESC LIMIT 1",
            (task_id,),
        ).fetchone()
        return json.loads(row[0]) if row and row[0] else []

    def drop_publish_copies(self, finalized_before: float, limit: int) -> int:
        """Clears the publish job kept with each verdict once it was finalized before `finalized_before`."""
        (mark,) = self.db.execute(
            "SELECT COALESCE((SELECT mark FROM retention WHERE name = 'final_verdicts'), 0)"
        ).fetchone()
        # Rows are inserted as uploads finalize, so row order is time order.
        rows = self.db.execute(
            "SELECT rowid, finalized_at FROM final_verdicts WHERE rowid > ? ORDER BY rowid LIMIT ?",
            (int(mark), limit),
        ).fetchall()
        last = int(mark)
        for rowid, at in rows:
            if at >= finalized_before:
                break
            last = rowid
        if last > mark:
            self.db.execute(
                "UPDATE final_verdicts SET publish = NULL"
                " WHERE rowid > ? AND rowid <= ? AND publish IS NOT NULL",
                (int(mark), last),
            )
            self.db.execute(
                "INSERT INTO retention VALUES ('final_verdicts', ?)"
                " ON CONFLICT (name) DO UPDATE SET mark = excluded.mark",
                (last,),
            )
        self.db.commit()
        return last - int(mark)

    def prune_urls(self, keep_days: float = URL_DETAIL_DAYS) -> None:
        self.db.execute(
            "UPDATE validations SET urls = NULL WHERE urls IS NOT NULL AND scored_at < ?",
            (time.time() - keep_days * 86400,),
        )
        self.db.commit()

    def recent(
        self,
        miner: str | None = None,
        validator: str | None = None,
        since: float = 0.0,
        before: float | None = None,
        limit: int = 50,
        verdict: str | None = None,
        kind: str | None = None,
    ) -> list[dict]:
        where, args = ["scored_at >= ?"], [since]
        if before is not None:
            where.append("scored_at < ?")
            args.append(before)
        for column, value in (
            ("miner", miner),
            ("validator", validator),
            ("verdict", verdict),
            ("kind", kind),
        ):
            if value:
                where.append(f"{column} = ?")
                args.append(value)
        rows = self.db.execute(
            f"SELECT {', '.join(FIELDS)} FROM validations WHERE {' AND '.join(where)}"
            " ORDER BY scored_at DESC LIMIT ?",
            (*args, limit),
        ).fetchall()
        return [dict(zip(FIELDS, row, strict=True)) for row in rows]

    def verdicts(self, miner: str | None = None) -> dict[str, int]:
        if miner is None:
            rows = self.db.execute(
                "SELECT verdict, SUM(n) FROM verdict_counts GROUP BY verdict"
            )
        else:
            rows = self.db.execute(
                "SELECT verdict, n FROM verdict_counts WHERE miner = ?", (miner,)
            )
        return {"pass": 0, "fail": 0, **dict(rows.fetchall())}

    def judged_since(self, miner: str, since: float, kind: str = "crawl") -> int:
        (count,) = self.db.execute(
            "SELECT COUNT(*) FROM validations WHERE miner = ? AND kind = ?"
            " AND scored_at >= ? AND verdict IN ('pass', 'fail')",
            (miner, kind, since),
        ).fetchone()
        return count

    def withdraw(
        self, miner: str, since: float, checked_too: bool = False
    ) -> list[tuple[str, int]]:
        """Passed uploads of a miner completed after `since`, taken back: (task_id, credited) of each; only those no validator checked unless `checked_too`."""
        rows = self.db.execute(
            "SELECT id, task_id, credited FROM validations"
            " WHERE miner = ? AND kind = 'crawl' AND verdict = 'pass' AND scored_at > ?"
            " AND COALESCE(completed_at, scored_at) > ? AND (? OR validator = '')",
            (miner, since, since, checked_too),
        ).fetchall()
        for row_id, _, _ in rows:
            self.db.execute(
                "UPDATE validations SET verdict = ? WHERE id = ?", (WITHDRAWN, row_id)
            )
        if rows:
            self.db.execute(
                "UPDATE verdict_counts SET n = n - ? WHERE miner = ? AND verdict = 'pass'",
                (len(rows), miner),
            )
            self.db.execute(
                "INSERT INTO verdict_counts VALUES (?, ?, ?)"
                " ON CONFLICT (miner, verdict) DO UPDATE SET n = n + excluded.n",
                (miner, WITHDRAWN, len(rows)),
            )
        self.db.commit()
        return [(task_id, credited) for _, task_id, credited in rows]

    def record_audit(self, agreed: list[str], disagreed: list[str]) -> None:
        for hotkey, disagreement in [(h, 0) for h in agreed] + [
            (h, 1) for h in disagreed
        ]:
            self.db.execute(
                "INSERT INTO validator_audits VALUES (?, 1, ?) ON CONFLICT (hotkey) DO UPDATE"
                " SET audits = audits + 1, disagreements = disagreements + excluded.disagreements",
                (hotkey, disagreement),
            )
        self.db.commit()

    def audit_standing(self) -> dict[str, dict]:
        rows = self.db.execute(
            "SELECT hotkey, audits, disagreements FROM validator_audits"
        ).fetchall()
        return {
            hotkey: {
                "audits": audits,
                "disagreements": disagreements,
                "excluded": audits >= MIN_AUDITS
                and disagreements / audits > MAX_DISAGREEMENT,
            }
            for hotkey, audits, disagreements in rows
        }

    def is_excluded(self, hotkey: str) -> bool:
        return self.audit_standing().get(hotkey, {}).get("excluded", False)

    def audits_of(self, hotkey: str) -> int:
        row = self.db.execute(
            "SELECT audits FROM validator_audits WHERE hotkey = ?", (hotkey,)
        ).fetchone()
        return row[0] if row else 0


VERDICT_SUMS = ", ".join(f"SUM(verdict = '{verdict}')" for verdict in VERDICTS)


def as_vote(row: tuple) -> dict:
    vote = dict(zip(VOTE_FIELDS, row, strict=True))
    vote["agreed"] = None if vote["agreed"] is None else bool(vote["agreed"])
    vote["decided"] = bool(vote["decided"])
    return vote


def utc_day(at: float | None = None) -> str:
    return datetime.fromtimestamp(at or time.time(), timezone.utc).strftime("%Y-%m-%d")


def build_report(
    task_id: str,
    job: dict,
    validator: str,
    result: dict,
    votes: list[dict] | None = None,
) -> dict:
    report = {
        **dict.fromkeys(COUNTS, 0),
        **{name: value for name, value in result.items() if name != "urls"},
        "task_id": task_id,
        "kind": job.get("kind", "crawl"),
        "round_id": job["round_id"],
        "miner": job["miner"],
        "validator": validator,
        "upload_key": job["key"],
        "upload_etag": job.get("etag", ""),
        "upload_bytes": job.get("size", 0),
        "page_key": None,
        "claimed_at": job.get("claimed_at"),
        "completed_at": job.get("completed_at"),
        "report_key": f"reports/dt={utc_day()}/task={task_id}.json",
        "scored_at": time.time(),
    }
    if votes and len(votes) > 1:
        report["votes"] = [
            {name: vote[name] for name in ("validator", "verdict", "credited")}
            | {"reason": vote["result"].get("reason", "")}
            for vote in votes
        ]
    return report
