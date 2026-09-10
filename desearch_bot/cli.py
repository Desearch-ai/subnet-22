"""desearch-bot command line."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import pyarrow.parquet as pq

from . import candidates, publish, qualify


def _language_detector():
    import py3langid

    def detect(text: str) -> str | None:
        return py3langid.classify(text)[0] if len(text) >= 20 else None

    return detect


def _pending(candidates_path: Path, done_path: Path, limit: int, order: str):
    table = pq.read_table(
        candidates_path, columns=["host", "rank", "tld_group", "excluded"]
    )
    rows = table.to_pylist()
    rows = [r for r in rows if not r["excluded"]]
    if order == "rank":
        rows.sort(key=lambda r: (r["rank"] is None, r["rank"] or 0, r["host"]))
    done = set()
    if done_path.exists():
        import json

        with open(done_path, encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    done.add(json.loads(line)["host"])
    pending = [
        (r["host"], r["rank"], r["tld_group"])
        for r in rows
        if r["host"] not in done
    ]
    return pending[:limit] if limit else pending, len(done)


def cmd_candidates(args):
    stats = candidates.build(
        Path(args.data_dir), Path(args.out), download=not args.no_download
    )
    print(f"{stats['total']:,} domains, {stats['kept']:,} candidates")
    for reason, count in stats["excluded"].items():
        print(f"  excluded {reason}: {count:,}")


def cmd_qualify(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    results_path = out / "results.jsonl"
    pending, already = _pending(
        Path(args.candidates), results_path, args.limit, args.order
    )
    print(
        f"{already:,} already checked, {len(pending):,} to check, "
        f"concurrency {args.concurrency}",
        file=sys.stderr,
    )
    if not pending:
        return

    buffer, counters = (
        [],
        {"done": 0, "qualified": 0, "start": datetime.now(timezone.utc)},
    )

    def on_result(result):
        buffer.append(result)
        counters["done"] += 1
        counters["qualified"] += result.qualified
        if len(buffer) >= 500:
            qualify.write_jsonl(results_path, buffer)
            buffer.clear()
            elapsed = (datetime.now(timezone.utc) - counters["start"]).total_seconds()
            print(
                f"  {counters['done']:,} checked, {counters['qualified']:,} qualified, "
                f"{counters['done'] / max(elapsed, 1):.0f}/s",
                file=sys.stderr,
                flush=True,
            )

    asyncio.run(
        qualify.qualify_hosts(
            pending, _language_detector(), args.concurrency, args.timeout, on_result
        )
    )
    if buffer:
        qualify.write_jsonl(results_path, buffer)
    print(f"{counters['done']:,} checked, {counters['qualified']:,} qualified")


def cmd_publish(args):
    import asyncio

    from . import db, publish

    async def run():
        pool = await db.connect(4)
        stats = await publish.export(pool, Path(args.out))
        await pool.close()
        return stats

    stats = asyncio.run(run())
    print(json.dumps(stats, indent=2))
    if not args.push:
        print("dry run; add --push to commit to the dataset")
        return
    token = os.environ.get("HF_TOKEN")
    if not token:
        sys.exit("HF_TOKEN is not set")
    publish.upload(Path(args.out), args.repo, token, stats)
    print(f"pushed to {args.repo} under domains/")


def cmd_initdb(args):
    import asyncio

    from . import db

    async def run():
        pool = await db.connect()
        await db.create_schema(pool)
        print(await db.counts(pool))
        await pool.close()

    asyncio.run(run())


def cmd_load(args):
    import asyncio

    import pyarrow.parquet as pq

    from . import db

    table = pq.read_table(args.candidates,
                          columns=["host", "rank", "tld_group", "excluded"])
    rows = [(r["host"], r["rank"], r["tld_group"])
            for r in table.to_pylist() if not r["excluded"]]
    if args.limit:
        rows = rows[: args.limit]

    async def run():
        pool = await db.connect()
        await db.create_schema(pool)
        for start in range(0, len(rows), 50_000):
            await db.load_candidates(pool, rows[start:start + 50_000])
            print(f"  loaded {min(start + 50_000, len(rows)):,}/{len(rows):,}", flush=True)
        print(await db.counts(pool))
        await pool.close()

    asyncio.run(run())


def cmd_discover(args):
    import asyncio

    from . import adult, db, discover, signing
    from .frontier import Frontier
    from .suffixes import PublicSuffixList

    async def run():
        pool = await db.connect(pool_size=args.pool)
        await db.create_schema(pool)
        rows = await db.take_candidates(pool, args.limit)
        print(f"{len(rows):,} candidates, concurrency {args.concurrency}", flush=True)
        hosts = [(r["host"], r["rank"], r["tld_group"]) for r in rows]
        adult_domains = adult.load(Path(args.data_dir), refresh=args.refresh_lists)
        print(f"{len(adult_domains):,} adult domains loaded", flush=True)
        signer = signing.from_env()
        print(f"signing keyid {signer.keyid}" if signer else "requests unsigned", flush=True)
        frontier = Frontier(Path(args.frontier))
        progress = discover.Progress()
        psl = PublicSuffixList(Path(args.data_dir) / "public_suffix_list.dat")
        await discover.discover(pool, frontier, hosts, _language_detector(),
                                args.concurrency, args.timeout, progress, adult_domains,
                                signer, psl)
        print("frontier:", frontier.stats())
        print(await db.counts(pool))
        await pool.close()

    asyncio.run(run())


def cmd_categorize(args):
    import asyncio

    from . import categories, db, exclusions

    async def run():
        pool = await db.connect(args.pool)
        await db.create_schema(pool)
        catalogue = categories.Catalogue.load(
            Path(args.data_dir), refresh=args.refresh_lists
        )
        print(
            f"{len(catalogue.by_label)} categories, "
            f"{sum(len(h) for h in catalogue.by_label.values()):,} labelled hosts",
            flush=True,
        )

        seen = saved = 0
        doomed: list[str] = []
        removed = {"domains": 0, "sitemaps": 0}
        async for rows in db.iter_hosts(pool):
            batch, rollup = [], []
            for record in rows:
                host = record["host"]
                labels = catalogue.labels(host)
                if labels:
                    rollup.append((host, labels, labels[0]))
                    batch.extend((host, "ut1", label, None, None, None) for label in labels)
                if catalogue.excluded(labels) or exclusions.exclusion_reason(
                    host, {}, record["tld_group"] or ""
                ) or exclusions.blocked_operator(host):
                    doomed.append(host)
            await db.save_domain_categories(pool, batch)
            saved += await db.save_category_rollup(pool, rollup)
            seen += len(rows)
            if args.prune and len(doomed) >= 50_000:
                for key, n in (await db.delete_hosts(pool, doomed)).items():
                    removed[key] += n
                doomed = []
            print(f"  {seen:,} scanned  {saved:,} labelled  "
                  f"{len(doomed) + removed['domains']:,} excluded", flush=True)

        if args.prune and doomed:
            for key, n in (await db.delete_hosts(pool, doomed)).items():
                removed[key] += n

        print(f"{seen:,} domains scanned, {saved:,} labelled")
        if args.prune:
            print(f"removed {removed['domains']:,} domains "
                  f"and {removed['sitemaps']:,} sitemap rows")
        else:
            print(f"{len(doomed):,} carry an excluded category; add --prune to remove them")
        print(await db.counts(pool))
        await pool.close()

    asyncio.run(run())


def cmd_resolve(args):
    import asyncio

    from . import db, reachability

    def report(counts):
        print("  %(checked)s checked  %(resolves)s resolve  %(dead)s dead"
              % {k: f"{v:,}" for k, v in counts.items()}, flush=True)

    async def run():
        pool = await db.connect(args.pool)
        counts = await reachability.resolve_all(
            pool, args.concurrency, args.rate, report
        )
        print("resolved %(checked)s: %(resolves)s live, %(dead)s dead"
              % {k: f"{v:,}" for k, v in counts.items()})
        await pool.close()

    asyncio.run(run())


def cmd_canonicalise(args):
    import asyncio

    from . import db, reachability, signing
    from .suffixes import PublicSuffixList

    def report(counts):
        print("  %(checked)s checked  %(redirects)s redirect  %(errors)s errors"
              % {k: f"{v:,}" for k, v in counts.items()}, flush=True)

    async def run():
        pool = await db.connect(args.pool)
        psl = PublicSuffixList(Path(args.data_dir) / "public_suffix_list.dat")
        counts = await reachability.canonicalise_all(
            pool, psl, args.concurrency, signing.from_env(), report
        )
        print("checked %(checked)s: %(redirects)s redirect elsewhere, %(errors)s errors"
              % {k: f"{v:,}" for k, v in counts.items()})
        await pool.close()

    asyncio.run(run())


def cmd_radar_categories(args):
    import asyncio

    from . import db, radar

    if args.top not in radar.BUCKETS:
        sys.exit(f"--top must be one of {', '.join(map(str, radar.BUCKETS))}")

    def report(counts):
        print("  %(checked)s checked  %(categorised)s categorised  %(failed)s failed"
              % {k: f"{v:,}" for k, v in counts.items()}, flush=True)

    async def run():
        path = Path(args.data_dir) / f"radar_top_{args.top}.csv"
        if not path.exists():
            radar.download_bucket(args.top, path)
        ranked = sorted(radar.read_bucket(path))
        pool = await db.connect(4)
        listed = await db.existing_hosts(pool, ranked)
        done = await db.checked_hosts(pool, "radar")
        todo = [host for host in ranked if host in listed and host not in done]
        print(f"{len(ranked):,} in Radar's top {args.top:,}; {len(listed):,} on our list; "
              f"{len(todo):,} still to look up", flush=True)
        counts = await radar.categorise(pool, todo, args.rate, on_batch=report)
        print("looked up %(checked)s: %(categorised)s with a category, %(failed)s failed"
              % {k: f"{v:,}" for k, v in counts.items()})
        await pool.close()

    asyncio.run(run())


def main(argv=None):
    parser = argparse.ArgumentParser(prog="desearch-bot")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("candidates", help="merge public lists into a candidate table")
    p.add_argument("--data-dir", default="data")
    p.add_argument("--out", default="build")
    p.add_argument("--no-download", action="store_true")
    p.set_defaults(func=cmd_candidates)

    p = sub.add_parser("qualify", help="visit candidates and keep the crawlable ones")
    p.add_argument("--candidates", default="build/candidates.parquet")
    p.add_argument("--out", default="build")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--concurrency", type=int, default=256)
    p.add_argument("--timeout", type=float, default=8.0)
    p.add_argument("--order", choices=["rank", "file"], default="rank")
    p.set_defaults(func=cmd_qualify)

    p = sub.add_parser(
        "publish", help="build the dataset files and optionally push them"
    )
    p.add_argument("--out", default="build")
    p.add_argument(
        "--repo", default=os.environ.get("HF_DATASET_REPO", "desearch/subnet-22")
    )
    p.add_argument("--push", action="store_true")
    p.set_defaults(func=cmd_publish)

    p = sub.add_parser("initdb", help="create the database schema")
    p.set_defaults(func=cmd_initdb)

    p = sub.add_parser("load", help="load candidate domains into the database")
    p.add_argument("--candidates", default="build/candidates.parquet")
    p.add_argument("--limit", type=int, default=0)
    p.set_defaults(func=cmd_load)

    p = sub.add_parser("resolve", help="DNS pass: flag which domains still resolve")
    p.add_argument("--concurrency", type=int, default=200)
    p.add_argument("--rate", type=float, default=300.0,
                   help="lookups per second; the NIC packet budget is the limit")
    p.add_argument("--pool", type=int, default=8)
    p.set_defaults(func=cmd_resolve)

    p = sub.add_parser("canonicalise", help="follow redirects and record the real domain")
    p.add_argument("--concurrency", type=int, default=200)
    p.add_argument("--pool", type=int, default=8)
    p.add_argument("--data-dir", default="data")
    p.set_defaults(func=cmd_canonicalise)

    p = sub.add_parser("radar-categories",
                       help="record Cloudflare Radar's categories for its top domains")
    p.add_argument("--top", type=int, default=10000)
    p.add_argument("--rate", type=float, default=3.5)
    p.add_argument("--data-dir", default="data")
    p.set_defaults(func=cmd_radar_categories)

    p = sub.add_parser("categorize", help="label domains and optionally drop excluded ones")
    p.add_argument("--data-dir", default="data")
    p.add_argument("--pool", type=int, default=8)
    p.add_argument("--prune", action="store_true",
                   help="delete domains carrying an excluded category")
    p.add_argument("--refresh-lists", action="store_true")
    p.set_defaults(func=cmd_categorize)

    p = sub.add_parser("discover", help="qualify domains and collect their sitemap URLs")
    p.add_argument("--limit", type=int, default=10000)
    p.add_argument("--concurrency", type=int, default=200)
    p.add_argument("--timeout", type=float, default=7.0)
    p.add_argument("--pool", type=int, default=16)
    p.add_argument("--frontier", default="/var/lib/desearch-bot/frontier")
    p.add_argument("--data-dir", default="data")
    p.add_argument("--refresh-lists", action="store_true",
                   help="re-download the adult blocklists before this run")
    p.set_defaults(func=cmd_discover)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
