"""Merge the public lists into one candidate table."""

from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from . import exclusions, sources
from .suffixes import PublicSuffixList, suffix, tld_group

SCHEMA = pa.schema(
    [
        ("host", pa.string()),
        ("suffix", pa.string()),
        ("tld_group", pa.string()),
        ("rank", pa.int32()),
        ("n_lists", pa.int8()),
        ("tranco_rank", pa.int32()),
        ("majestic_rank", pa.int32()),
        ("opr_rank", pa.int32()),
        ("builtwith_rank", pa.int32()),
        ("umbrella_rank", pa.int32()),
        ("excluded", pa.bool_()),
        ("exclude_reason", pa.string()),
        ("flags", pa.list_(pa.string())),
        ("type_hint", pa.string()),
    ]
)


def build(data_dir: Path, out_dir: Path, download: bool = True) -> dict:
    data_dir, out_dir = Path(data_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = (
        sources.fetch_all(data_dir)
        if download
        else {name: data_dir / f"{name}.bin" for name in sources.RANKED}
        | {"ut1": data_dir / "ut1.tar.gz", "psl": data_dir / "public_suffix_list.dat"}
    )

    psl = PublicSuffixList(paths["psl"])
    ranked = {
        name: sources.read_ranked(paths[name], psl, *spec[1:])
        for name, spec in sources.RANKED.items()
    }
    wanted = (
        set(exclusions.UT1_EXCLUDE)
        | set(exclusions.UT1_FLAG)
        | set(exclusions.UT1_TYPE_HINT)
    )
    categories = sources.read_categories(paths["ut1"], wanted)

    merged: dict[str, dict[str, int]] = {}
    for name, ranks in ranked.items():
        for host, rank in ranks.items():
            merged.setdefault(host, {})[name] = rank

    rows, reasons = [], Counter()
    for host, host_ranks in merged.items():
        traffic = [host_ranks[n] for n in sources.TRAFFIC_RANKED if n in host_ranks]
        group = tld_group(host)
        reason = exclusions.exclusion_reason(host, categories, group)
        reasons[reason or "kept"] += 1
        rows.append(
            {
                "host": host,
                "suffix": suffix(host),
                "tld_group": group,
                "rank": min(traffic) if traffic else None,
                "n_lists": len(host_ranks),
                "tranco_rank": host_ranks.get("tranco"),
                "majestic_rank": host_ranks.get("majestic"),
                "opr_rank": host_ranks.get("opr"),
                "builtwith_rank": host_ranks.get("builtwith"),
                "umbrella_rank": host_ranks.get("umbrella"),
                "excluded": reason is not None,
                "exclude_reason": reason,
                "flags": [
                    c for c in exclusions.UT1_FLAG if host in categories.get(c, ())
                ],
                "type_hint": next(
                    (
                        v
                        for c, v in exclusions.UT1_TYPE_HINT.items()
                        if host in categories.get(c, ())
                    ),
                    None,
                ),
            }
        )
    rows.sort(
        key=lambda r: (r["rank"] is None, r["rank"] or 0, -r["n_lists"], r["host"])
    )

    built_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    table = pa.Table.from_pylist(rows, schema=SCHEMA).replace_schema_metadata(
        {
            b"built_at": built_at.encode(),
            b"generator": b"desearch_bot.candidates",
        }
    )
    pq.write_table(
        table,
        out_dir / "candidates.parquet",
        compression="zstd",
        row_group_size=200_000,
    )

    stats = {
        "built_at": built_at,
        "sources": {name: len(r) for name, r in ranked.items()},
        "total": len(rows),
        "kept": reasons["kept"],
        "excluded": dict(
            sorted(
                ((k, v) for k, v in reasons.items() if k != "kept"),
                key=lambda kv: -kv[1],
            )
        ),
    }
    (out_dir / "candidates_stats.json").write_text(json.dumps(stats, indent=2))
    return stats
