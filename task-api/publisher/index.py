"""The publisher's own record of every page's latest version and of the tasks taken back."""

from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass

WITHDRAWN_KEEP_S = 7 * 86400


@dataclass
class Current:
    url: str
    version: str
    fetched_at: str
    task_id: str
    content_sha1: str
    change_seq: int | None = None
    change_row: int | None = None


class VersionIndex:
    def __init__(self, path: str):
        self.db = sqlite3.connect(path, check_same_thread=False)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.executescript(
            """
            CREATE TABLE IF NOT EXISTS pages (
                key          TEXT PRIMARY KEY,
                url          TEXT NOT NULL,
                version      TEXT NOT NULL,
                fetched_at   TEXT NOT NULL,
                task_id      TEXT NOT NULL,
                content_sha1 TEXT NOT NULL
            ) WITHOUT ROWID;
            CREATE INDEX IF NOT EXISTS pages_task ON pages (task_id);
            CREATE TABLE IF NOT EXISTS withdrawn (
                task_id TEXT PRIMARY KEY,
                at      REAL NOT NULL
            ) WITHOUT ROWID;
            """
        )
        # Where the newest version's full record sits: the change file's number and its row.
        have = {row[1] for row in self.db.execute("PRAGMA table_info(pages)")}
        for column in ("change_seq", "change_row"):
            if column not in have:
                self.db.execute(f"ALTER TABLE pages ADD COLUMN {column} INTEGER")
        self.db.commit()

    def current(self, key: str) -> Current | None:
        row = self.db.execute(
            "SELECT url, version, fetched_at, task_id, content_sha1, change_seq,"
            " change_row FROM pages WHERE key = ?",
            (key,),
        ).fetchone()
        return Current(*row) if row else None

    def store(self, records: list[dict]) -> None:
        """Called only after the change file holding these records is written."""
        self.db.executemany(
            "INSERT INTO pages (key, url, version, fetched_at, task_id, content_sha1,"
            " change_seq, change_row) VALUES (?, ?, ?, ?, ?, ?, ?, ?)"
            " ON CONFLICT (key) DO UPDATE SET"
            " url = excluded.url, version = excluded.version,"
            " fetched_at = excluded.fetched_at, task_id = excluded.task_id,"
            " content_sha1 = excluded.content_sha1, change_seq = excluded.change_seq,"
            " change_row = excluded.change_row",
            [
                (
                    r["key"],
                    r["url"],
                    r["version"],
                    r["fetched_at"],
                    r["task_id"],
                    r["content_sha1"],
                    r.get("change_seq"),
                    r.get("change_row"),
                )
                for r in records
            ],
        )
        self.db.commit()

    def touch(self, seen: list[tuple[str, str]]) -> None:
        """A later fetch that found the same content moves the fetch time forward."""
        self.db.executemany(
            "UPDATE pages SET fetched_at = MAX(fetched_at, ?) WHERE key = ?",
            [(fetched_at, key) for key, fetched_at in seen],
        )
        self.db.commit()

    def of_tasks(self, task_ids: list[str]) -> list[tuple[str, str, str, str]]:
        """Pages whose current version came from these tasks: (key, url, version, content_sha1)."""
        found = []
        for task_id in task_ids:
            found += self.db.execute(
                "SELECT key, url, version, content_sha1 FROM pages WHERE task_id = ?",
                (task_id,),
            ).fetchall()
        return found

    def withdraw(self, task_ids: list[str]) -> None:
        """Remembered, so a withdrawn task's pages are never published even if its job comes later."""
        now = time.time()
        self.db.executemany(
            "INSERT OR IGNORE INTO withdrawn VALUES (?, ?)",
            [(task_id, now) for task_id in task_ids],
        )
        self.db.execute("DELETE FROM withdrawn WHERE at < ?", (now - WITHDRAWN_KEEP_S,))
        self.db.commit()

    def is_withdrawn(self, task_id: str) -> bool:
        found = self.db.execute(
            "SELECT 1 FROM withdrawn WHERE task_id = ?", (task_id,)
        ).fetchone()
        return found is not None

    def remove(self, removed: list[tuple[str, str]]) -> None:
        """A withdrawal removes a page only while the withdrawn version is still current."""
        self.db.executemany("DELETE FROM pages WHERE key = ? AND version = ?", removed)
        self.db.commit()

    def backup_to(self, path: str) -> None:
        target = sqlite3.connect(path)
        with target:
            self.db.backup(target)
        target.close()

    def close(self) -> None:
        self.db.close()
