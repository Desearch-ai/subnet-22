"""Keep bulk sources current: run a source's commands, publish only its new and changed pages, hide the pages it dropped."""

from __future__ import annotations

import argparse
import fcntl
import glob
import hashlib
import json
import os
import sqlite3
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import pyarrow.parquet as pq

from bulk.rows import MANIFEST, REMOVED, Batches

HOME_ENV = "REFRESH_HOME"
DEFAULT_HOME = "~/refresh"
OUTBOX = "outbox"
RECEIPTS = "receipts"
READY = ".ready"
RECEIPT_POLL_S = 30
HIDE_SSH = "refresh-hide"
STATUS = "status.jsonl"
MIN_SEEN_SHARE = 0.95
LARGE_SHARE = 0.30
COMMAND_TIMEOUT_S = 6 * 3600
PUBLISH_TIMEOUT_S = 4 * 3600
REPO = Path(__file__).resolve().parents[1]
UNHASHED = frozenset({"fetched_at", "elapsed_ms", "html", "html_bytes", "html_sha256"})


@dataclass(frozen=True)
class Build:
    """One `python -m bulk` command whose run folder is diffed and published on its own."""

    name: str
    argv: tuple[str, ...]
    full_list: bool = True


@dataclass(frozen=True)
class Source:
    name: str
    fetch: tuple[tuple[str, ...], ...]
    builds: tuple[Build, ...]
    cwd: str = ""


def bulk_build(command: str, *inputs: str) -> tuple[str, ...]:
    return (
        "{python}",
        "-m",
        "bulk",
        command,
        "--out",
        "{out}",
        "--as-of",
        "{as_of}",
        "--run",
        "{run}",
        *inputs,
    )


BITTENSOR = Source(
    "bittensor",
    fetch=(("{python}", "-m", "bulk", "bittensor-fetch", "--out", "{work}/src"),),
    builds=(
        Build("subnets", bulk_build("subnets", "{work}/src/subnets-chain.json")),
        Build("subnet-sites", bulk_build("subnet-sites", "{work}/src/sites")),
        Build("subnet-github", bulk_build("subnet-github", "{work}/src/github")),
    ),
)
ARXIV = Source(
    "arxiv",
    fetch=(("{python}", "-m", "bulk", "arxiv-harvest", "--out", "{work}/oai", "--from", "{since}", "--until", "{until}"),),
    builds=(Build("arxiv", bulk_build("arxiv", "{work}/oai/*.xml"), full_list=False),),
)
SOURCES = {source.name: source for source in (BITTENSOR, ARXIV)}


class Refused(RuntimeError):
    """A run that publishes and hides nothing, because its input looks broken."""


@dataclass
class Counts:
    seen: int = 0
    new: int = 0
    changed: int = 0
    unchanged: int = 0
    removed: int = 0
    known: int = 0
    published: str = ""
    large: bool = False
    hashes: dict[str, str] = field(default_factory=dict, repr=False)
    gone: list[str] = field(default_factory=list, repr=False)

    def report(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if k not in ("hashes", "gone")}


def home() -> Path:
    return Path(os.environ.get(HOME_ENV, DEFAULT_HOME)).expanduser()


def page_hash(row: dict) -> str:
    kept = {k: v for k, v in row.items() if k not in UNHASHED}
    return hashlib.sha256(
        json.dumps(kept, sort_keys=True, default=str).encode()
    ).hexdigest()


def open_state(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(path)
    con.execute(
        "CREATE TABLE IF NOT EXISTS pages (url TEXT PRIMARY KEY, build TEXT NOT NULL, hash TEXT NOT NULL,"
        " run TEXT NOT NULL, seen TEXT NOT NULL, removed TEXT)"
    )
    return con


def rows_of(folder: Path):
    manifest = json.loads((folder / MANIFEST).read_text())
    for name in manifest["files"]:
        yield from pq.read_table(folder / name).to_pylist()


def removed_of(folder: Path) -> list[str]:
    path = folder / REMOVED
    return [u for u in path.read_text().split() if u] if path.exists() else []


def diff(
    con: sqlite3.Connection,
    build: Build,
    folder: Path,
    out: Path,
    run: str,
    as_of: datetime,
) -> Counts:
    """Writes the build's new and changed pages to `out` as a run of their own, and counts what changed."""
    known = dict(
        con.execute(
            "SELECT url, hash FROM pages WHERE build = ? AND removed IS NULL",
            (build.name,),
        )
    )
    source = json.loads((folder / MANIFEST).read_text())["source"]
    counts = Counts(known=len(known))
    batches = Batches(out, source, run, as_of)
    for row in rows_of(folder):
        url = row["url"]
        if url in counts.hashes:
            continue
        counts.hashes[url] = h = page_hash(row)
        before = known.get(url)
        if before == h:
            counts.unchanged += 1
            continue
        if before is None:
            counts.new += 1
        else:
            counts.changed += 1
        batches.add(row)
    counts.seen = len(counts.hashes)
    if build.full_list and known and counts.seen < MIN_SEEN_SHARE * len(known):
        raise Refused(
            f"{build.name}: {counts.seen} pages, under {MIN_SEEN_SHARE:.0%} of the {len(known)} known"
        )
    dropped = [u for u in known if u not in counts.hashes] if build.full_list else []
    counts.gone = sorted(
        set(dropped) | {u for u in removed_of(folder) if u not in counts.hashes}
    )
    counts.removed = len(counts.gone)
    counts.large = bool(known) and counts.new + counts.changed > LARGE_SHARE * len(
        known
    )
    batches.close({"pages": batches.rows, "new": counts.new, "changed": counts.changed})
    return counts


def keep(
    con: sqlite3.Connection, build: Build, counts: Counts, run: str, as_of: datetime
) -> None:
    """Records a published diff, so the next run compares with it."""
    stamp = as_of.strftime("%Y-%m-%dT%H:%M:%SZ")
    with con:
        con.executemany(
            "INSERT INTO pages (url, build, hash, run, seen, removed) VALUES (?, ?, ?, ?, ?, NULL)"
            " ON CONFLICT(url) DO UPDATE SET build = excluded.build, seen = excluded.seen, removed = NULL,"
            " run = CASE WHEN pages.hash = excluded.hash AND pages.removed IS NULL THEN pages.run ELSE excluded.run END,"
            " hash = excluded.hash",
            ((url, build.name, h, run, stamp) for url, h in counts.hashes.items()),
        )
        con.executemany(
            "UPDATE pages SET removed = ? WHERE url = ? AND build = ?",
            ((run, url, build.name) for url in counts.gone),
        )


def hidden_urls(con: sqlite3.Connection) -> list[str]:
    return [
        u
        for (u,) in con.execute(
            "SELECT url FROM pages WHERE removed IS NOT NULL ORDER BY url"
        )
    ]


def expand(part: str, values: dict[str, str]) -> list[str]:
    """One argument filled in; a path with a wildcard becomes the files it matches."""
    filled = part.format(python=sys.executable, **values)
    if filled.startswith("/") and any(c in filled for c in "*?["):
        return sorted(glob.glob(filled)) or [filled]
    return [filled]


def run_command(argv: tuple[str, ...], values: dict[str, str], log: Path, cwd: Path = REPO) -> None:
    command = [arg for part in argv for arg in expand(part, values)]
    with log.open("a") as out:
        out.write(f"$ {' '.join(command)}\n")
        out.flush()
        subprocess.run(
            command,
            cwd=cwd,
            stdout=out,
            stderr=subprocess.STDOUT,
            check=True,
            timeout=COMMAND_TIMEOUT_S,
        )


def publish(folder: Path, root: Path) -> str:
    """Puts a run folder in the outbox the planner box pulls from, and waits for its receipt; returns the publish line."""
    outbox, receipts = root / OUTBOX, root / RECEIPTS
    outbox.mkdir(parents=True, exist_ok=True)
    receipts.mkdir(parents=True, exist_ok=True)
    os.rename(folder, outbox / folder.name)
    (outbox / f"{folder.name}{READY}").touch()
    receipt = receipts / f"{folder.name}.txt"
    deadline = time.time() + PUBLISH_TIMEOUT_S
    while not receipt.exists():
        if time.time() > deadline:
            raise RuntimeError(
                f"{folder.name}: no receipt from the planner box in {PUBLISH_TIMEOUT_S // 3600} h"
            )
        time.sleep(RECEIPT_POLL_S)
    lines = receipt.read_text().splitlines()
    if "rc=0" not in lines:
        raise RuntimeError(
            f"publishing {folder.name} failed: {' | '.join(lines[-5:])[:600]}"
        )
    published = [line for line in lines if "bulk files" in line]
    return published[-1] if published else lines[-1]


def hide(source: str, urls: list[str]) -> None:
    """Rewrites the source's section of index-1's hidden-URLs file to exactly these URLs."""
    done = subprocess.run(
        ["ssh", "-o", "BatchMode=yes", HIDE_SSH, source],
        input="".join(f"{u}\n" for u in urls),
        capture_output=True,
        text=True,
        timeout=300,
    )
    if done.returncode:
        raise RuntimeError(
            f"hiding {source}'s removed pages failed: {done.stderr[-400:]}"
        )


def last_ok(root: Path, source: str) -> str:
    """The as-of time of the source's last run that published, or ""."""
    path, last = root / STATUS, ""
    if path.exists():
        for line in path.read_text().splitlines():
            entry = json.loads(line)
            if entry.get("source") == source and entry.get("ok") and not entry.get("dry_run"):
                last = max(last, entry.get("as_of", ""))
    return last


def refresh(
    source: Source,
    root: Path,
    dry_run: bool = False,
    now: datetime | None = None,
    since: str = "",
) -> dict:
    as_of = (now or datetime.now(timezone.utc)).replace(microsecond=0)
    since = since or last_ok(root, source.name)[:10]
    if not since and any("{since}" in part for argv in source.fetch for part in argv):
        raise Refused(f"{source.name} has no run that published yet; give its first run --since")
    run = as_of.strftime("%Y%m%d-%H%M")
    work = root / "work" / source.name / run
    work.mkdir(parents=True, exist_ok=False)
    log = work / "log.txt"
    values = {
        "work": str(work),
        "as_of": as_of.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "run": run,
        "since": since,
        "until": as_of.strftime("%Y-%m-%d"),
    }
    cwd = Path(source.cwd).expanduser() if source.cwd else REPO
    for argv in source.fetch:
        run_command(argv, values, log, cwd)
    con = open_state(root / "state" / f"{source.name}.sqlite")
    diffs: dict[str, tuple[Build, Counts, Path]] = {}
    for build in source.builds:
        folder = work / "build" / build.name
        run_command(build.argv, {**values, "out": str(folder)}, log, cwd)
        out = work / "publish" / f"refresh-{build.name}-{run}"
        out.parent.mkdir(parents=True, exist_ok=True)
        diffs[build.name] = (build, diff(con, build, folder, out, run, as_of), out)
    report = {"source": source.name, "run": run, "as_of": values["as_of"], "since": since, "dry_run": dry_run, "builds": {}}
    for name, (build, counts, out) in diffs.items():
        if not dry_run:
            if counts.new + counts.changed:
                counts.published = publish(out, root)
            keep(con, build, counts, run, as_of)
        report["builds"][name] = counts.report()
    if not dry_run:
        hide(source.name, hidden_urls(con))
    report["hidden"] = len(hidden_urls(con))
    return report


def seed(con: sqlite3.Connection, build: Build, folders: list[Path]) -> int:
    """Records runs published before the runner, so its first diff sends only what changed since."""
    pages = 0
    for folder in folders:
        manifest = json.loads((folder / MANIFEST).read_text())
        as_of = datetime.fromisoformat(manifest["as_of"].replace("Z", "+00:00"))
        counts = Counts(hashes={row["url"]: page_hash(row) for row in rows_of(folder)})
        keep(con, build, counts, manifest["run"], as_of)
        pages += len(counts.hashes)
    return pages


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m bulk.refresh", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    one = commands.add_parser("run", help="refresh one source now")
    one.add_argument("source", choices=sorted(SOURCES))
    one.add_argument(
        "--dry-run",
        action="store_true",
        help="fetch, build and diff, but publish and hide nothing",
    )
    one.add_argument("--since", default="", help="first day to fetch, YYYY-MM-DD; the last published run's day by default")
    first = commands.add_parser(
        "seed", help="record runs already published, before a source's first refresh"
    )
    first.add_argument("source", choices=sorted(SOURCES))
    first.add_argument("build")
    first.add_argument("folders", type=Path, nargs="+")
    a = parser.parse_args(argv)
    root = home()
    source = SOURCES[a.source]
    if a.command == "seed":
        build = next((b for b in source.builds if b.name == a.build), None)
        if build is None:
            parser.error(
                f"{a.source} has no build {a.build}; it has {', '.join(b.name for b in source.builds)}"
            )
        pages = seed(
            open_state(root / "state" / f"{source.name}.sqlite"), build, a.folders
        )
        print(f"{source.name}/{build.name}: {pages} pages recorded")
        return 0
    root.mkdir(parents=True, exist_ok=True)
    started = time.time()
    with (root / "lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            report = refresh(source, root, a.dry_run, since=a.since)
            report["ok"] = True
        except Exception as e:
            report = {
                "source": source.name,
                "ok": False,
                "refused": isinstance(e, Refused),
                "error": repr(e)[:1000],
            }
        report["seconds"] = round(time.time() - started)
        report["finished"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        with (root / STATUS).open("a") as status:
            status.write(json.dumps(report) + "\n")
    print(json.dumps(report, indent=1))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
