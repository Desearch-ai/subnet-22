from __future__ import annotations

import json
import sqlite3

from .rounds import Batch, Round, Url

COLUMNS = (
    "round_id, kind, manifest_hash, seed_block, opened_at, seed, serve_order,"
    " closed_at, batches"
)


class RoundStore:
    def __init__(self, db: sqlite3.Connection):
        self.db = db
        self.db.executescript(
            """
            CREATE TABLE IF NOT EXISTS rounds (
                round_id      TEXT PRIMARY KEY,
                kind          TEXT NOT NULL,
                manifest_hash TEXT NOT NULL,
                seed_block    INTEGER NOT NULL,
                opened_at     REAL NOT NULL,
                seed          TEXT,
                serve_order   TEXT NOT NULL DEFAULT '[]',
                closed_at     REAL,
                batches       TEXT NOT NULL,
                filled_at     REAL
            );
            CREATE INDEX IF NOT EXISTS rounds_opened ON rounds (opened_at);
            CREATE INDEX IF NOT EXISTS rounds_pending ON rounds (seed, closed_at);
            """
        )
        self.db.commit()

    def save(self, round_: Round) -> None:
        batches = {
            batch_id: {
                "urls": [[u.host, u.url] for u in batch.urls],
                "extra": batch.extra,
            }
            for batch_id, batch in round_.batches.items()
        }
        self.db.execute(
            f"INSERT INTO rounds ({COLUMNS}) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)"
            " ON CONFLICT (round_id) DO UPDATE SET seed = excluded.seed,"
            " serve_order = excluded.serve_order, closed_at = excluded.closed_at",
            (
                round_.round_id,
                round_.kind,
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
            f"SELECT {COLUMNS} FROM rounds WHERE round_id = ?", (round_id,)
        ).fetchone()
        return _round(row) if row else None

    def unrevealed(self) -> list[Round]:
        rows = self.db.execute(
            f"SELECT {COLUMNS} FROM rounds WHERE seed IS NULL ORDER BY opened_at"
        ).fetchall()
        return [_round(row) for row in rows]

    def unfilled(self) -> list[Round]:
        """Revealed rounds whose tasks never reached the queue."""
        rows = self.db.execute(
            f"SELECT {COLUMNS} FROM rounds WHERE seed IS NOT NULL"
            " AND filled_at IS NULL AND closed_at IS NULL ORDER BY opened_at"
        ).fetchall()
        return [_round(row) for row in rows]

    def mark_filled(self, round_id: str, at: float) -> None:
        self.db.execute(
            "UPDATE rounds SET filled_at = ? WHERE round_id = ?", (at, round_id)
        )
        self.db.commit()

    def open_revealed(self) -> list[str]:
        rows = self.db.execute(
            "SELECT round_id FROM rounds WHERE seed IS NOT NULL"
            " AND filled_at IS NOT NULL AND closed_at IS NULL"
        ).fetchall()
        return [row[0] for row in rows]

    def latest_revealed(self) -> dict[str, str]:
        """The newest revealed round of each kind that is still open."""
        rows = self.db.execute(
            "SELECT kind, round_id FROM rounds WHERE seed IS NOT NULL"
            " AND closed_at IS NULL ORDER BY opened_at"
        ).fetchall()
        return dict(rows)

    def close(self, round_id: str, at: float) -> None:
        self.db.execute(
            "UPDATE rounds SET closed_at = ? WHERE round_id = ?", (at, round_id)
        )
        self.db.commit()

    def recent(self, limit: int = 100) -> list[dict]:
        rows = self.db.execute(
            "SELECT round_id, kind, manifest_hash, seed_block, opened_at,"
            " seed IS NOT NULL, closed_at FROM rounds ORDER BY opened_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [
            {
                "round_id": round_id,
                "kind": kind,
                "manifest_hash": manifest_hash,
                "seed_block": seed_block,
                "opened_at": opened_at,
                "revealed": bool(revealed),
                "closed_at": closed_at,
            }
            for (
                round_id,
                kind,
                manifest_hash,
                seed_block,
                opened_at,
                revealed,
                closed_at,
            ) in rows
        ]


def _round(row: tuple) -> Round:
    (
        round_id,
        kind,
        manifest_hash,
        seed_block,
        opened_at,
        seed,
        order,
        closed_at,
        batches,
    ) = row
    return Round(
        round_id=round_id,
        batches={
            batch_id: Batch(
                batch_id,
                [Url(host, url) for host, url in stored["urls"]],
                stored["extra"],
            )
            for batch_id, stored in json.loads(batches).items()
        },
        manifest_hash=manifest_hash,
        seed_block=seed_block,
        opened_at=opened_at,
        kind=kind,
        seed=seed,
        order=json.loads(order),
        closed_at=closed_at,
    )
