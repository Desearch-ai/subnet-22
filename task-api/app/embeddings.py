from __future__ import annotations

import sqlite3
import time

QUEUED, DONE, DROPPED = "queued", "done", "dropped"


class Embeddings:
    """Which version of each page has vectors, and from which model."""

    def __init__(self, db: sqlite3.Connection):
        self.db = db
        self.db.executescript(
            """
            CREATE TABLE IF NOT EXISTS embeddings (
                page_key     TEXT NOT NULL,
                model        TEXT NOT NULL,
                content_sha1 TEXT NOT NULL,
                url          TEXT NOT NULL,
                state        TEXT NOT NULL,
                batch_id     TEXT NOT NULL,
                vectors_key  TEXT,
                updated_at   REAL NOT NULL,
                PRIMARY KEY (page_key, model)
            );
            CREATE INDEX IF NOT EXISTS embeddings_model_state ON embeddings (model, state);
            """
        )
        self.db.commit()

    def missing(self, pages: list[dict], model: str) -> list[dict]:
        """The pages whose current text this model has not embedded or queued yet."""
        wanted = []
        for page in pages:
            row = self.db.execute(
                "SELECT content_sha1, state FROM embeddings"
                " WHERE page_key = ? AND model = ?",
                (page["page_key"], model),
            ).fetchone()
            if row is None or row[0] != page["content_sha1"] or row[1] == DROPPED:
                wanted.append(page)
        return wanted

    def queue(self, pages: list[dict], model: str, batch_id: str) -> None:
        self.db.executemany(
            "INSERT INTO embeddings"
            " (page_key, model, content_sha1, url, state, batch_id, updated_at)"
            " VALUES (?, ?, ?, ?, ?, ?, ?)"
            " ON CONFLICT (page_key, model) DO UPDATE SET"
            " content_sha1 = excluded.content_sha1, url = excluded.url,"
            " state = excluded.state, batch_id = excluded.batch_id,"
            " vectors_key = NULL, updated_at = excluded.updated_at",
            [
                (
                    p["page_key"],
                    model,
                    p["content_sha1"],
                    p["url"],
                    QUEUED,
                    batch_id,
                    time.time(),
                )
                for p in pages
            ],
        )
        self.db.commit()

    def settle(
        self, batch_id: str, model: str, state: str, vectors_key: str | None = None
    ) -> int:
        """Only rows still queued by this batch change; a newer version keeps its own state."""
        changed = self.db.execute(
            "UPDATE embeddings SET state = ?, vectors_key = ?, updated_at = ?"
            " WHERE batch_id = ? AND model = ? AND state = ?",
            (state, vectors_key, time.time(), batch_id, model, QUEUED),
        ).rowcount
        self.db.commit()
        return changed

    def of_page(self, page_key: str) -> list[dict]:
        rows = self.db.execute(
            "SELECT model, content_sha1, url, state, batch_id, vectors_key, updated_at"
            " FROM embeddings WHERE page_key = ? ORDER BY model",
            (page_key,),
        ).fetchall()
        names = (
            "model",
            "content_sha1",
            "url",
            "state",
            "batch_id",
            "vectors_key",
            "updated_at",
        )
        return [dict(zip(names, row, strict=True)) for row in rows]

    def counts(self) -> dict[str, dict[str, int]]:
        counted: dict[str, dict[str, int]] = {}
        for model, state, n in self.db.execute(
            "SELECT model, state, COUNT(*) FROM embeddings GROUP BY model, state"
        ):
            counted.setdefault(model, {})[state] = n
        return counted
