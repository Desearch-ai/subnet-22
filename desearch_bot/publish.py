"""Publish the domain list to the public dataset.

One column, one row per domain. Everything operational — sitemaps, crawl delay, refresh
schedule, categories — stays in the database; miners only need to know which hosts are in scope.
The file is the database's domain table, so the two always carry the same set.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from . import db

SCHEMA = pa.schema([("host", pa.string())])
CARD = Path(__file__).with_name("dataset_card.md")
STALE = ("domains/stats.json",)


async def export(pool, out_dir: Path) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "domains.parquet"

    total = 0
    writer = pq.ParquetWriter(path, SCHEMA, compression="zstd")
    try:
        async for rows in db.iter_hosts(pool):
            hosts = pa.array([r["host"] for r in rows], pa.string())
            writer.write_table(pa.table({"host": hosts}, SCHEMA))
            total += len(rows)
    finally:
        writer.close()

    built_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    (out_dir / "README.md").write_text(card(total, built_at))
    return {"built_at": built_at, "domains": total, "bytes": path.stat().st_size}


def card(total: int, built_at: str) -> str:
    return CARD.read_text().replace("{{DOMAINS}}", f"{total:,}").replace(
        "{{BUILT_AT}}", built_at
    )


def upload(out_dir: Path, repo: str, token: str, stats: dict) -> None:
    """Replace the published list, dropping files an earlier layout left behind."""
    from huggingface_hub import CommitOperationAdd, CommitOperationDelete, HfApi

    out_dir = Path(out_dir)
    api = HfApi(token=token)
    present = set(api.list_repo_files(repo, repo_type="dataset"))
    operations = [
        CommitOperationAdd("domains/domains.parquet", str(out_dir / "domains.parquet")),
        CommitOperationAdd("README.md", str(out_dir / "README.md")),
    ]
    operations += [CommitOperationDelete(p) for p in STALE if p in present]

    api.create_commit(
        repo_id=repo,
        repo_type="dataset",
        operations=operations,
        commit_message=f"domains: {stats['domains']:,} domains",
    )
