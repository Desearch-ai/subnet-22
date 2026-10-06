"""The Python publisher on the sample against a pre-filled index: `prefill SAMPLE WORK PAGES`, then `run SAMPLE WORK PROCESSES [PASSES]`."""

from __future__ import annotations

import hashlib
import json
import multiprocessing
import os
import resource
import shutil
import statistics
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import publisher.worker as worker
from publisher.index import VersionIndex
from publisher.records import record_key
from reference import NOW, Clock, Pages, Queue, Temp, seed_for

CHUNK = 100_000
_reader = None


def cpu() -> float:
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return usage.ru_utime + usage.ru_stime


def peak() -> int:
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rss if sys.platform == "darwin" else rss * 1024


def synthetic(i: int) -> dict:
    """A page another task published earlier, shaped like the Rust benchmark's."""
    domain = f"news{i % 20011}.example{i % 7}.com"
    digest = hashlib.sha1(i.to_bytes(8, "little")).hexdigest()
    url = f"https://www.{domain}/2026/10/{i:08d}-{digest[:24]}-story-about-something"
    fetched = datetime.fromtimestamp(NOW - i % 2_592_000, timezone.utc)
    return {
        "key": f"pages/{domain}/{hashlib.sha1(url.encode()).hexdigest()}",
        "url": url,
        "version": hashlib.sha1(f"v{i}".encode()).hexdigest(),
        "fetched_at": fetched.isoformat(timespec="seconds"),
        "task_id": f"{((i // 1000) * 0x9E3779B97F4A7C15) % (1 << 64):016x}",
        "content_sha1": hashlib.sha1(f"c{i}".encode()).hexdigest(),
        "change_seq": i // 20000,
        "change_row": i % 20000,
    }


def reader(sample: Path, jobs: list[dict]) -> worker.Publisher:
    worker.datetime = Clock
    return worker.Publisher(Queue(), Temp(sample, jobs), Pages(), workers=1)


def start_reader(sample: str) -> None:
    global _reader
    jobs = json.loads((Path(sample) / "jobs.json").read_text())
    _reader = reader(Path(sample), jobs)


def warm(_):
    time.sleep(0.3)
    return os.getpid()


def collect_share(jobs: list[dict]):
    """Runs in a reader process: each job's records and missed URLs, with the CPU, memory and bytes it took."""
    before, read, requests = cpu(), _reader.temp.bytes, _reader.temp.requests
    found = []
    for job in jobs:
        try:
            found.append(_reader.collect(job))
        except Exception as exc:
            found.append(RuntimeError(repr(exc)))
    return found, cpu() - before, (os.getpid(), peak()), _reader.temp.bytes - read, _reader.temp.requests - requests


def prefill(sample: Path, work: Path, count: int) -> None:
    jobs = json.loads((sample / "jobs.json").read_text())
    seeding = reader(sample, jobs)
    seed = seed_for([record for job in jobs for record in seeding.collect(job)[0]])
    seeding.close()
    path = work / "py-template.sqlite"
    for suffix in ("", "-wal", "-shm"):
        Path(f"{path}{suffix}").unlink(missing_ok=True)
    started = time.monotonic()
    index = VersionIndex(str(path))
    index.store(seed)
    for start in range(0, count, CHUNK):
        index.store([synthetic(i) for i in range(start, min(start + CHUNK, count))])
    index.db.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    index.close()
    total = count + len(seed)
    print(
        json.dumps(
            {
                "pages": total,
                "seeded": len(seed),
                "seconds": time.monotonic() - started,
                "file_bytes_per_page": path.stat().st_size / total,
            }
        )
    )


def run(sample: Path, work: Path, processes: int, passes: int) -> None:
    jobs = json.loads((sample / "jobs.json").read_text())
    path = work / f"py-run-{processes}.sqlite"
    shutil.copyfile(work / "py-template.sqlite", path)
    index = VersionIndex(str(path))
    main = reader(sample, jobs)
    main.index = index
    shares = [jobs[n::processes] for n in range(processes)]
    with ProcessPoolExecutor(
        processes,
        mp_context=multiprocessing.get_context("spawn"),
        initializer=start_reader,
        initargs=(str(sample),),
    ) as pool:
        list(pool.map(warm, range(processes * 2)))
        walls, cpus = [], []
        for _ in range(max(passes, 1)):
            before, started = cpu(), time.monotonic()
            done = list(pool.map(collect_share, shares))
            walls.append(time.monotonic() - started)
            cpus.append(cpu() - before + sum(part[1] for part in done))
        reader_peaks = {}
        for part in done:
            pid, rss = part[2]
            reader_peaks[pid] = max(rss, reader_peaks.get(pid, 0))
    found = [None] * len(jobs)
    for n, (part, *_rest) in enumerate(done):
        for i, outcome in zip(range(n, len(jobs), processes), part):
            found[i] = outcome
    bytes_in = sum(part[3] for part in done)
    requests = sum(part[4] for part in done)

    before, started = cpu(), time.monotonic()
    chosen, failed = {}, []
    for outcome in found:
        if isinstance(outcome, BaseException):
            continue
        records, missed = outcome
        for record in records:
            key = record_key(record)
            if key not in chosen or worker._rank(record) > worker._rank(chosen[key]):
                chosen[key] = record
        failed += missed
    changes, unchanged = main.decide(chosen)
    key = main.write_changes(changes)
    body = main.pages.objects[key]
    index.store([worker._indexed(change, 1, row) for row, change in enumerate(changes)])
    index.touch([(record_key(record), record["fetched_at"]) for record in unchanged])
    write_wall, write_cpu = time.monotonic() - started, cpu() - before
    read_wall, read_cpu = statistics.median(walls), statistics.median(cpus)
    tasks = len(jobs)
    new = sum(1 for c in changes if c["kind"] == "new")
    print(
        json.dumps(
            {
                "processes": processes,
                "tasks": tasks,
                "not_read": sum(1 for f in found if isinstance(f, BaseException)),
                "pages_new": new,
                "pages_changed": len(changes) - new,
                "pages_unchanged": len(unchanged),
                "urls_failed": len(failed),
                "read_wall_s": read_wall,
                "read_cpu_s": read_cpu,
                "write_wall_s": write_wall,
                "write_cpu_s": write_cpu,
                "cpu_s_per_task": (read_cpu + write_cpu) / tasks,
                "read_cpu_s_per_task": read_cpu / tasks,
                "write_cpu_s_per_task": write_cpu / tasks,
                "wall_s_per_task": (read_wall + write_wall) / tasks,
                "tasks_per_min_per_core": 60 * tasks / (read_cpu + write_cpu),
                "tasks_per_min_pipelined": 60 * tasks / max(read_wall, write_wall),
                "peak_rss_main_bytes": peak(),
                "peak_rss_total_bytes": peak() + sum(reader_peaks.values()),
                "r2_bytes_in_per_task": bytes_in / tasks,
                "r2_requests_per_task": requests / tasks,
                "change_file_bytes": len(body),
                "change_file_bytes_per_page": len(body) / max(len(changes), 1),
            }
        )
    )
    index.close()
    for suffix in ("", "-wal", "-shm"):
        Path(f"{path}{suffix}").unlink(missing_ok=True)


if __name__ == "__main__":
    command, sample, work = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3])
    work.mkdir(parents=True, exist_ok=True)
    if command == "prefill":
        prefill(sample, work, int(sys.argv[4]))
    else:
        run(
            sample, work, int(sys.argv[4]), int(sys.argv[5]) if len(sys.argv) > 5 else 3
        )
