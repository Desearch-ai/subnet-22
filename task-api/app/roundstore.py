from __future__ import annotations

import json
import sqlite3

from .rounds import Batch, Round, Url


class RoundStore:
    def __init__(self, db: sqlite3.Connection):
        self.db = db
        self.db.executescript(
            """
            CREATE TABLE IF NOT EXISTS rounds (
                round_id      TEXT PRIMARY KEY,
                manifest_hash TEXT NOT NULL,
                seed_block    INTEGER NOT NULL,
                opened_at     REAL NOT NULL,
                seed          TEXT,
                serve_order   TEXT NOT NULL DEFAULT '[]',
                closed_at     REAL,
                batches       TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS rounds_opened ON rounds (opened_at);
            CREATE INDEX IF NOT EXISTS rounds_pending ON rounds (seed, closed_at);
            """
        )
        self.db.commit()

    def save(self, round_: Round) -> None:
        batches = {
            batch_id: [[u.host, u.url] for u in batch.urls]
            for batch_id, batch in round_.batches.items()
        }
        self.db.execute(
            "INSERT OR REPLACE INTO rounds VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                round_.round_id,
                round_.manifest_hash,
                round_.seed_block,
                round_.opened_at,
                round_.seed,
                json.dumps(round_.order),
                round_.closed_at,
                json.dumps(batches),
            ),
        )
        self.db.commit()

    def get(self, round_id: str) -> Round | None:
        row = self.db.execute(
            "SELECT * FROM rounds WHERE round_id = ?", (round_id,)
        ).fetchone()
        return _round(row) if row else None

    def unrevealed(self) -> list[Round]:
        rows = self.db.execute(
            "SELECT * FROM rounds WHERE seed IS NULL ORDER BY opened_at"
        ).fetchall()
        return [_round(row) for row in rows]

    def open_revealed(self) -> list[str]:
        rows = self.db.execute(
            "SELECT round_id FROM rounds WHERE seed IS NOT NULL AND closed_at IS NULL"
        ).fetchall()
        return [row[0] for row in rows]

    def latest_revealed(self) -> str | None:
        row = self.db.execute(
            "SELECT round_id FROM rounds WHERE seed IS NOT NULL"
            " ORDER BY opened_at DESC LIMIT 1"
        ).fetchone()
        return row[0] if row else None

    def close(self, round_id: str, at: float) -> None:
        self.db.execute(
            "UPDATE rounds SET closed_at = ? WHERE round_id = ?", (at, round_id)
        )
        self.db.commit()

    def recent(self, limit: int = 100) -> list[dict]:
        rows = self.db.execute(
            "SELECT round_id, manifest_hash, seed_block, opened_at, seed IS NOT NULL,"
            " closed_at FROM rounds ORDER BY opened_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [
            {
                "round_id": round_id,
                "manifest_hash": manifest_hash,
                "seed_block": seed_block,
                "opened_at": opened_at,
                "revealed": bool(revealed),
                "closed_at": closed_at,
            }
            for round_id, manifest_hash, seed_block, opened_at, revealed, closed_at in rows
        ]


def _round(row: tuple) -> Round:
    round_id, manifest_hash, seed_block, opened_at, seed, order, closed_at, batches = (
        row
    )
    return Round(
        round_id=round_id,
        batches={
            batch_id: Batch(batch_id, [Url(host, url) for host, url in urls])
            for batch_id, urls in json.loads(batches).items()
        },
        manifest_hash=manifest_hash,
        seed_block=seed_block,
        opened_at=opened_at,
        seed=seed,
        order=json.loads(order),
        closed_at=closed_at,
    )
