"""Publish the qualified domains from the database to the public dataset."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

SCHEMA = pa.schema(
    [
        ("host", pa.string()),
        ("sitemap_url", pa.string()),
        ("sitemap_kind", pa.string()),
        ("url_count", pa.int64()),
        ("crawl_delay", pa.float32()),
    ]
)

CARD = Path(__file__).with_name("dataset_card.md")


async def export(pool, out_dir: Path) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    async with pool.acquire() as connection:
        rows = await connection.fetch(
            """
            SELECT host, sitemap_url, sitemap_kind, url_count, crawl_delay
            FROM bot.domains
            WHERE status = 'qualified'
            ORDER BY host
            """
        )
        rejected = await connection.fetch(
            """
            SELECT reject_reason, count(*) AS n
            FROM bot.domains WHERE status = 'rejected'
            GROUP BY 1 ORDER BY 2 DESC
            """
        )
        by_group = await connection.fetch(
            """
            SELECT tld_group, count(*) AS n
            FROM bot.domains WHERE status = 'qualified'
            GROUP BY 1 ORDER BY 2 DESC
            """
        )
        totals = await connection.fetchrow(
            """
            SELECT count(*) FILTER (WHERE status = 'qualified') AS qualified,
                   count(*) FILTER (WHERE status = 'rejected') AS rejected,
                   count(*) FILTER (WHERE status = 'candidate') AS remaining,
                   coalesce(sum(url_count), 0) AS urls,
                   (SELECT count(*) FROM bot.sitemaps) AS sitemaps
            FROM bot.domains
            """
        )

    table = pa.Table.from_pylist([dict(r) for r in rows], schema=SCHEMA)
    built_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    table = table.replace_schema_metadata({b"built_at": built_at.encode()})
    pq.write_table(
        table, out_dir / "domains.parquet", compression="zstd", row_group_size=100_000
    )

    stats = {
        "built_at": built_at,
        "qualified": int(totals["qualified"]),
        "rejected": int(totals["rejected"]),
        "still_to_check": int(totals["remaining"]),
        "urls_discovered": int(totals["urls"]),
        "sitemaps_tracked": int(totals["sitemaps"]),
        "qualified_by_tld_group": {r["tld_group"]: int(r["n"]) for r in by_group},
        "rejected_by_reason": {
            r["reject_reason"] or "unknown": int(r["n"]) for r in rejected
        },
    }
    (out_dir / "stats.json").write_text(json.dumps(stats, indent=2))
    return stats


def upload(out_dir: Path, repo: str, token: str, stats: dict) -> None:
    from huggingface_hub import CommitOperationAdd, HfApi

    out_dir = Path(out_dir)
    operations = [
        CommitOperationAdd("domains/domains.parquet", str(out_dir / "domains.parquet")),
        CommitOperationAdd("domains/stats.json", str(out_dir / "stats.json")),
    ]
    if CARD.exists():
        operations.append(CommitOperationAdd("README.md", str(CARD)))

    HfApi(token=token).create_commit(
        repo_id=repo,
        repo_type="dataset",
        operations=operations,
        commit_message=(
            f"domains: {stats['qualified']:,} crawlable domains, "
            f"{stats['urls_discovered']:,} URLs discovered"
        ),
    )
