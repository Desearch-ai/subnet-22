"""Compares the Python and Rust publishers' dumps field by field: compare.py WORK_DIR."""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow.parquet as pq

EXAMPLES = 3


class Tally:
    def __init__(self):
        self.compared = Counter()
        self.differ = Counter()
        self.examples = defaultdict(list)

    def field(self, where: str, name: str, ours, theirs, context: str = "") -> None:
        label = f"{where}.{name}"
        self.compared[label] += 1
        if ours != theirs:
            self.differ[label] += 1
            if len(self.examples[label]) < EXAMPLES:
                self.examples[label].append(
                    {"at": context, "rust": ours, "python": theirs}
                )

    def rows(self, where: str, rust: list[dict], python: list[dict], key=None) -> None:
        self.field(where, "#rows", len(rust), len(python))
        for i, (r, p) in enumerate(zip(rust, python)):
            context = (key(p) if key else str(i))[:160]
            for name in sorted(set(r) | set(p)):
                self.field(
                    where,
                    name,
                    r.get(name, "<absent>"),
                    p.get(name, "<absent>"),
                    context,
                )

    def report(self) -> int:
        width = max(len(label) for label in self.compared)
        for label in sorted(self.compared):
            n, bad = self.compared[label], self.differ[label]
            print(
                f"{label:<{width}}  {n:>7} compared  {n - bad:>7} equal  {100 * (n - bad) / n:7.3f}%"
            )
        for label, examples in sorted(self.examples.items()):
            print(f"\n{label} differs, e.g.:")
            for example in examples:
                print("  " + json.dumps(example, ensure_ascii=False)[:600])
        return sum(self.differ.values())


def by_task(jobs: list[dict]) -> dict:
    return {job["task_id"]: job for job in jobs}


def main() -> None:
    work = Path(sys.argv[1])
    python = json.loads((work / "python.json").read_text())
    rust = json.loads((work / "rust.json").read_text())
    tally = Tally()
    for number, (p, r) in enumerate(zip(python["batches"], rust["batches"]), 1):
        where = f"batch{number}"
        pj, rj = by_task(p["jobs"]), by_task(r["jobs"])
        tally.field(where, "jobs", sorted(rj), sorted(pj))
        for task_id in pj:
            pjob, rjob = pj[task_id], rj.get(task_id, {})
            tally.field(
                where, "job_read_ok", "records" in rjob, "records" in pjob, task_id
            )
            if "records" in pjob and "records" in rjob:
                tally.rows(
                    f"{where}.record",
                    rjob["records"],
                    pjob["records"],
                    key=lambda x: x["assigned_url"],
                )
                tally.field(
                    where, "job_failed", rjob["failed"], pjob["failed"], task_id
                )
        tally.rows(
            f"{where}.change", r["changes"], p["changes"], key=lambda x: x["key"]
        )
        tally.rows(
            f"{where}.unchanged", r["unchanged"], p["unchanged"], key=lambda x: x["key"]
        )
        tally.rows(f"{where}.stored", r["stored"], p["stored"], key=lambda x: x["key"])
        tally.field(where, "touched", r["touched"], p["touched"])
        tally.field(where, "removed", r["removed"], p["removed"])
        tally.rows(
            f"{where}.outcome", r["outcomes"], p["outcomes"], key=lambda x: x["url"]
        )
        tally.field(where, "lost", r["lost"], p["lost"])
        tally.field(where, "acked", r["acked"], p["acked"])
        tally.field(
            where, "has_change_file", bool(r["change_file"]), bool(p["change_file"])
        )
        if r["change_file"] and p["change_file"]:
            rf, pf = pq.ParquetFile(r["change_file"]), pq.ParquetFile(p["change_file"])
            tally.field(
                where, "file.schema", str(rf.schema_arrow), str(pf.schema_arrow)
            )
            tally.field(
                where,
                "file.schema_equal",
                rf.schema_arrow.equals(pf.schema_arrow),
                True,
            )
            tally.field(
                where,
                "file.row_groups",
                [rf.metadata.row_group(g).num_rows for g in range(rf.num_row_groups)],
                [pf.metadata.row_group(g).num_rows for g in range(pf.num_row_groups)],
            )
            tally.rows(
                f"{where}.file",
                rf.read().to_pylist(),
                pf.read().to_pylist(),
                key=lambda x: x["key"] or "",
            )
    pages_p, pages_r = python["index"]["pages"], rust["index"]["pages"]
    tally.field("index", "keys", sorted(pages_r), sorted(pages_p))
    for key in sorted(set(pages_p) & set(pages_r)):
        for name in pages_p[key]:
            tally.field("index", name, pages_r[key].get(name), pages_p[key][name], key)
    tally.field(
        "index",
        "withdrawn",
        sorted(rust["index"]["withdrawn"]),
        python["index"]["withdrawn"],
    )
    sys.exit(1 if tally.report() else 0)


if __name__ == "__main__":
    main()
