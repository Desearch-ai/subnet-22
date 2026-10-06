"""The publisher's own record of every page's latest version and of the tasks taken back."""

from __future__ import annotations

import sqlite3
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

WITHDRAWN_KEEP_S = 7 * 86400
CACHE_KIB = 1 << 20
# The index outgrows memory, so each lookup is a random disk read; many readers at once keep the disk busy.
READERS = 4
READ_CHUNK = 500
CURRENT_COLUMNS = (
    "url, version, fetched_at, task_id, content_sha1, change_seq, change_row"
)


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
        self.path = path
        self.local = threading.local()
        self.readers = None if path == ":memory:" else ThreadPoolExecutor(READERS)
        self.db = sqlite3.connect(path, check_same_thread=False)
        self.db.execute("PRAGMA journal_mode=WAL")
        # A crash loses at most the last batches, which replay; every batch is tens of thousands of upserts.
        self.db.execute("PRAGMA synchronous=NORMAL")
        self.db.execute(f"PRAGMA cache_size=-{CACHE_KIB}")
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
            f"SELECT {CURRENT_COLUMNS} FROM pages WHERE key = ?", (key,)
        ).fetchone()
        return Current(*row) if row else None

    def current_many(self, keys: list[str]) -> dict[str, Current]:
        """The current version of each page the index holds, looked up in sorted chunks on parallel readers."""
        keys = sorted(keys)
        chunks = [keys[i : i + READ_CHUNK] for i in range(0, len(keys), READ_CHUNK)]
        found: dict[str, Current] = {}
        reading = (
            self.readers.map(self.read_chunk, chunks)
            if self.readers
            else map(lambda chunk: self.read_chunk(chunk, self.db), chunks)
        )
        for part in reading:
            found.update(part)
        return found

    def read_chunk(
        self, keys: list[str], db: sqlite3.Connection | None = None
    ) -> dict[str, Current]:
        db = db or self.reader()
        rows = db.execute(
            f"SELECT key, {CURRENT_COLUMNS} FROM pages"
            f" WHERE key IN ({','.join('?' * len(keys))})",
            keys,
        ).fetchall()
        return {row[0]: Current(*row[1:]) for row in rows}

    def reader(self) -> sqlite3.Connection:
        """This thread's read-only connection."""
        if getattr(self.local, "db", None) is None:
            self.local.db = sqlite3.connect(
                f"file:{self.path}?mode=ro", uri=True, check_same_thread=False
            )
            self.local.db.execute(f"PRAGMA cache_size=-{CACHE_KIB // READERS}")
        return self.local.db

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
                for r in sorted(records, key=lambda r: r["key"])
            ],
        )
        self.db.commit()

    def touch(self, seen: list[tuple[str, str]]) -> None:
        """A later fetch that found the same content moves the fetch time forward."""
        self.db.executemany(
            "UPDATE pages SET fetched_at = MAX(fetched_at, ?) WHERE key = ?",
            [(fetched_at, key) for key, fetched_at in sorted(seen)],
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
        if self.readers is not None:
            self.readers.shutdown(wait=True)
        self.db.close()
