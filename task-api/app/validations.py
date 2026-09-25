from __future__ import annotations

import json
import sqlite3
import time
from collections import Counter
from dataclasses import dataclass, field
from datetime import UTC, datetime

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
    "scored_at",
)
MIN_AUDITS = 10
URL_DETAIL_DAYS = 7
MAX_DISAGREEMENT = 0.3
MAX_CREDIT_DIVERGENCE = 0.15
# The validator's own timeout or crash, never held against the miner.
VALIDATOR_FAULT_REASONS = frozenset({"unscorable"})
CONTENT_OUTCOMES = ("matched", "mismatched", "unverifiable", "not_fetched")
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
            "reason": "validators_disagree",
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

    def latest(self, task_id: str) -> dict | None:
        row = self.db.execute(
            f"SELECT {', '.join(FIELDS)} FROM validations"
            " WHERE task_id = ? ORDER BY id DESC LIMIT 1",
            (task_id,),
        ).fetchone()
        return dict(zip(FIELDS, row, strict=True)) if row else None

    def urls(self, task_id: str) -> list[dict]:
        row = self.db.execute(
            "SELECT urls FROM validations WHERE task_id = ? ORDER BY id DESC LIMIT 1",
            (task_id,),
        ).fetchone()
        return json.loads(row[0]) if row and row[0] else []

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
    ) -> list[dict]:
        where, args = ["scored_at >= ?"], [since]
        if before is not None:
            where.append("scored_at < ?")
            args.append(before)
        for column, value in (("miner", miner), ("validator", validator)):
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


def utc_day(at: float | None = None) -> str:
    return datetime.fromtimestamp(at or time.time(), UTC).strftime("%Y-%m-%d")


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
