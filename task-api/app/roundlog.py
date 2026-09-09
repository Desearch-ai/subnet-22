from __future__ import annotations

import json
import sqlite3
import time

from . import ordering


class RoundLog:
    def __init__(self, path: str = "roundlog.db"):
        self.db = sqlite3.connect(path, check_same_thread=False)
        self.db.executescript(
            """
            CREATE TABLE IF NOT EXISTS entries (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                round_id     TEXT NOT NULL,
                hotkey       TEXT NOT NULL,
                requested_at REAL NOT NULL,
                served_at    REAL NOT NULL,
                outcome      TEXT NOT NULL,
                task_id      TEXT,
                refusal      TEXT,
                receipt_sig  TEXT NOT NULL,
                seq          INTEGER NOT NULL DEFAULT 0
            );
            CREATE INDEX IF NOT EXISTS entries_round ON entries (round_id, id);
            CREATE TABLE IF NOT EXISTS anchors (
                round_id TEXT PRIMARY KEY,
                root     TEXT NOT NULL,
                at       REAL NOT NULL
            );
            """
        )
        self.db.commit()

    def record(
        self,
        round_id: str,
        hotkey: str,
        requested_at: float,
        outcome: str,
        receipt_sig: str,
        task_id: str | None = None,
        refusal: dict | None = None,
        seq: int = 0,
    ) -> dict:
        served_at = time.time()
        self.db.execute(
            "INSERT INTO entries (round_id, hotkey, requested_at, served_at, outcome, task_id,"
            " refusal, receipt_sig, seq) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                round_id,
                hotkey,
                requested_at,
                served_at,
                outcome,
                task_id,
                json.dumps(refusal) if refusal else None,
                receipt_sig,
                seq,
            ),
        )
        self.db.commit()
        return {"outcome": outcome, "task_id": task_id, "served_at": served_at}

    def entries(self, round_id: str) -> list[dict]:
        rows = self.db.execute(
            "SELECT hotkey, requested_at, served_at, outcome, task_id, refusal, receipt_sig, seq"
            " FROM entries WHERE round_id = ? ORDER BY seq, id",
            (round_id,),
        ).fetchall()
        out = []
        for hotkey, requested_at, served_at, outcome, task_id, refusal, sig, seq in rows:
            entry = {
                "hotkey": hotkey,
                "requested_at": requested_at,
                "served_at": served_at,
                "outcome": outcome,
                "receipt_sig": sig,
                "seq": seq,
            }
            if task_id:
                entry["task_id"] = task_id
            if refusal:
                entry["refusal"] = json.loads(refusal)
            out.append(entry)
        return out

    def anchor(self, round_id: str) -> str:
        root = ordering.merkle_root(
            [ordering.log_leaf(e) for e in self.entries(round_id)]
        )
        self.db.execute(
            "INSERT OR REPLACE INTO anchors (round_id, root, at) VALUES (?, ?, ?)",
            (round_id, root, time.time()),
        )
        self.db.commit()
        return root

    def anchored_root(self, round_id: str) -> str | None:
        row = self.db.execute(
            "SELECT root FROM anchors WHERE round_id = ?", (round_id,)
        ).fetchone()
        return row[0] if row else None
