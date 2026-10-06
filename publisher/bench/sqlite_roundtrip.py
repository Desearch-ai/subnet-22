"""A Python-made SQLite index through `publisher import-sqlite` and `export-sqlite`, compared row for row: sqlite_roundtrip.py BINARY WORK_DIR [PAGES]."""

from __future__ import annotations

import shutil
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

from publisher.index import VersionIndex


def rows(path: Path) -> tuple[list, list]:
    db = sqlite3.connect(path)
    pages = db.execute(
        "SELECT key, url, version, fetched_at, task_id, content_sha1, change_seq, change_row FROM pages ORDER BY key"
    ).fetchall()
    withdrawn = db.execute(
        "SELECT task_id, at FROM withdrawn ORDER BY task_id"
    ).fetchall()
    db.close()
    return pages, withdrawn


def main() -> None:
    binary, work = sys.argv[1], Path(sys.argv[2])
    count = int(sys.argv[3]) if len(sys.argv) > 3 else 200_000
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)
    original = work / "python.sqlite"
    index = VersionIndex(str(original))
    entries = [
        {
            "key": f"pages/site{i % 97}.com/{i:040x}",
            "url": f"https://www.site{i % 97}.com/story/{i}?q=é",
            "version": f"{i * 7919 % (1 << 160):040x}",
            "fetched_at": f"2026-10-{i % 28 + 1:02d}T{i % 24:02d}:00:00+00:00",
            "task_id": f"{i // 1000:016x}",
            "content_sha1": f"{i * 104729 % (1 << 160):040x}",
            "change_seq": None if i % 3 == 0 else i // 20000,
            "change_row": None if i % 3 == 0 else i % 20000,
        }
        for i in range(count)
    ]
    for start in range(0, count, 50_000):
        index.store(entries[start : start + 50_000])
    index.withdraw([f"{t:016x}" for t in range(5)])
    index.close()

    started = time.monotonic()
    subprocess.run(
        [binary, "import-sqlite", str(original), str(work / "rocks")], check=True
    )
    subprocess.run(
        [binary, "export-sqlite", str(work / "rocks"), str(work / "back.sqlite")],
        check=True,
    )
    seconds = time.monotonic() - started

    back = VersionIndex(str(work / "back.sqlite"))
    keys = [entry["key"] for entry in entries[:: max(count // 1000, 1)]]
    reopened = VersionIndex(str(original))
    assert back.current_many(keys) == reopened.current_many(keys)
    assert back.of_tasks(["0000000000000002"]) == reopened.of_tasks(
        ["0000000000000002"]
    )
    assert back.is_withdrawn("0000000000000004") and not back.is_withdrawn(
        "0000000000000009"
    )
    back.close()
    reopened.close()
    assert rows(original) == rows(work / "back.sqlite"), (
        "pages or withdrawn rows differ"
    )
    print(
        f"{count} pages and 5 withdrawn tasks through RocksDB and back unchanged, in {seconds:.1f}s"
    )


if __name__ == "__main__":
    main()
