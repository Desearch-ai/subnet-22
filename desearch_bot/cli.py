"""desearch-bot command line."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

from . import candidates


def _language_detector():
    import py3langid

    def detect(text: str) -> str | None:
        return py3langid.classify(text)[0] if len(text) >= 20 else None

    return detect


def cmd_candidates(args):
    stats = candidates.build(
        Path(args.data_dir), Path(args.out), download=not args.no_download
    )
    print(f"{stats['total']:,} domains, {stats['kept']:,} candidates")
    for reason, count in stats["excluded"].items():
        print(f"  excluded {reason}: {count:,}")


def cmd_publish(args):
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
    from . import db

    async def run():
        pool = await db.connect()
        await db.create_schema(pool)
        print(await db.counts(pool))
        await pool.close()

    asyncio.run(run())


def cmd_load(args):
    import pyarrow.parquet as pq

    from . import db

    table = pq.read_table(
        args.candidates, columns=["host", "rank", "tld_group", "excluded"]
    )
    rows = [
        (r["host"], r["rank"], r["tld_group"])
        for r in table.to_pylist()
        if not r["excluded"]
    ]
    if args.limit:
        rows = rows[: args.limit]

    async def run():
        pool = await db.connect()
        await db.create_schema(pool)
        for start in range(0, len(rows), 50_000):
            await db.load_candidates(pool, rows[start : start + 50_000])
            print(
                f"  loaded {min(start + 50_000, len(rows)):,}/{len(rows):,}", flush=True
            )
        print(await db.counts(pool))
        await pool.close()

    asyncio.run(run())


def cmd_categorize(args):
    from . import categories, db, exclusions

    async def run():
        pool = await db.connect(args.pool)
        catalogue = categories.Catalogue.load(
            Path(args.data_dir), refresh=args.refresh_lists
        )
        print(
            f"{len(catalogue.by_label)} categories, "
            f"{sum(len(h) for h in catalogue.by_label.values()):,} labelled hosts",
            flush=True,
        )
        seen = labelled = excluded = 0
        async for rows in db.iter_hosts(pool):
            batch, rollup, flagged = [], [], []
            for record in rows:
                host = record["host"]
                labels = catalogue.labels(host)
                if labels:
                    rollup.append((host, labels, labels[0]))
                    batch.extend(
                        (host, "ut1", label, None, None, None) for label in labels
                    )
                reason = catalogue.excluded(labels) or exclusions.exclusion_reason(
                    host, {}, record["tld_group"] or ""
                )
                if reason:
                    flagged.append((host, reason))
            await db.save_domain_categories(pool, batch)
            labelled += await db.save_category_rollup(pool, rollup)
            excluded += await db.exclude_hosts(pool, flagged)
            seen += len(rows)
            print(
                f"  {seen:,} scanned  {labelled:,} labelled  {excluded:,} newly excluded",
                flush=True,
            )
        print(
            f"{seen:,} domains scanned, {labelled:,} labelled, {excluded:,} newly excluded"
        )
        print(await db.counts(pool))
        await pool.close()

    asyncio.run(run())


def cmd_radar_categories(args):
    from . import db, radar

    if args.top not in radar.BUCKETS:
        sys.exit(f"--top must be one of {', '.join(map(str, radar.BUCKETS))}")

    def report(counts):
        print(
            "  %(checked)s checked  %(categorised)s categorised  %(failed)s failed"
            % {k: f"{v:,}" for k, v in counts.items()},
            flush=True,
        )

    async def run():
        path = Path(args.data_dir) / f"radar_top_{args.top}.csv"
        if not path.exists():
            radar.download_bucket(args.top, path)
        ranked = sorted(radar.read_bucket(path))
        pool = await db.connect(4)
        listed = await db.existing_hosts(pool, ranked)
        done = await db.checked_hosts(pool, "radar")
        todo = [host for host in ranked if host in listed and host not in done]
        print(
            f"{len(ranked):,} in Radar's top {args.top:,}; {len(listed):,} on our list; "
            f"{len(todo):,} still to look up",
            flush=True,
        )
        counts = await radar.categorise(pool, todo, args.rate, on_batch=report)
        print(
            "looked up %(checked)s: %(categorised)s with a category, %(failed)s failed"
            % {k: f"{v:,}" for k, v in counts.items()}
        )
        await pool.close()

    asyncio.run(run())


def cmd_run(args):
    import multiprocessing
    import signal

    workers = args.workers or max(1, (os.cpu_count() or 2) - 1)
    context = multiprocessing.get_context("spawn")
    processes = [
        context.Process(
            target=_worker, args=(index, workers, vars(args)), name=f"w{index}"
        )
        for index in range(workers)
    ]
    for process in processes:
        process.start()

    def stop(*_):
        for process in processes:
            if process.is_alive():
                process.terminate()

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    failed = False
    while any(process.is_alive() for process in processes):
        for process in processes:
            process.join(timeout=1)
            if process.exitcode not in (None, 0) and not failed:
                print(
                    f"worker {process.name} exited with {process.exitcode}; stopping",
                    flush=True,
                )
                failed = True
                stop()
    sys.exit(1 if failed else 0)


def _worker(index: int, workers: int, options: dict) -> None:
    try:
        import uvloop

        factory = uvloop.new_event_loop
    except ImportError:
        factory = None
    asyncio.run(
        _crawl(index, workers, argparse.Namespace(**options)), loop_factory=factory
    )


async def _crawl(index: int, workers: int, args) -> None:
    import signal
    import time

    from . import db, loop, net, signing
    from .buckets import Buckets, Resources, owned_buckets
    from .registry import Registry
    from .suffixes import PublicSuffixList
    from .visit import Visitor

    tag, started, printed = f"[w{index}]", time.monotonic(), [0.0]
    totals = {"requests": 0, "new": 0}

    def report(crawl, writes):
        totals["requests"] += sum(write.visit.requests for write in writes)
        totals["new"] += sum(write.visit.new for write in writes)
        now = time.monotonic()
        if now - printed[0] < 30:
            return
        printed[0] = now
        elapsed = max(now - started, 1)
        print(
            f"{tag} {crawl.visited:,} visited  {crawl.visited / elapsed:.1f}/s  "
            f"{totals['requests'] / elapsed:.1f} req/s  in flight {len(crawl.inflight)}  "
            f"new urls {totals['new']:,}  scheduled {len(crawl.timetable):,}",
            flush=True,
        )

    pool = await db.connect(args.pool)
    excluded = await db.excluded_categories(pool)
    psl = PublicSuffixList(Path(args.data_dir) / "public_suffix_list.dat")
    resources = Resources(args.cache_mb << 20, args.memtable_mb << 20)
    owned = owned_buckets(index, workers)
    with Buckets(Path(args.buckets_dir), owned, resources) as buckets:
        async with net.session(args.concurrency) as session:
            visitor = Visitor(
                session,
                buckets,
                _language_detector(),
                psl.registrable,
                signing.from_env(),
            )
            registry = Registry(pool, buckets)
            crawl = loop.Loop(
                buckets, visitor, args.concurrency, registry, excluded, report
            )
            print(f"{tag} {crawl.load():,} domains in {len(owned)} buckets", flush=True)
            for sig in (signal.SIGTERM, signal.SIGINT):
                asyncio.get_running_loop().add_signal_handler(sig, crawl.stop)
            await crawl.run()
            # A cancelled visit can leave a URL write running in a thread; let it land.
            await asyncio.get_running_loop().shutdown_default_executor()
    await pool.close()


def main(argv=None):
    parser = argparse.ArgumentParser(prog="desearch-bot")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("candidates", help="merge public lists into a candidate table")
    p.add_argument("--data-dir", default="data")
    p.add_argument("--out", default="build")
    p.add_argument("--no-download", action="store_true")
    p.set_defaults(func=cmd_candidates)

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

    p = sub.add_parser(
        "run", help="the crawl loop: one process per core, each owning its buckets"
    )
    p.add_argument(
        "--workers", type=int, default=0, help="crawler processes; 0 means cores - 1"
    )
    p.add_argument(
        "--concurrency", type=int, default=200, help="sites read at once, per worker"
    )
    p.add_argument("--pool", type=int, default=2)
    p.add_argument("--buckets-dir", default="/var/lib/desearch-bot/buckets")
    p.add_argument("--data-dir", default="data")
    p.add_argument("--cache-mb", type=int, default=512)
    p.add_argument("--memtable-mb", type=int, default=256)
    p.set_defaults(func=cmd_run)

    p = sub.add_parser(
        "categorize", help="label domains from offline lists and flag the excluded ones"
    )
    p.add_argument("--data-dir", default="data")
    p.add_argument("--pool", type=int, default=8)
    p.add_argument("--refresh-lists", action="store_true")
    p.set_defaults(func=cmd_categorize)

    p = sub.add_parser(
        "radar-categories",
        help="record Cloudflare Radar's categories for its top domains",
    )
    p.add_argument("--top", type=int, default=10000)
    p.add_argument("--rate", type=float, default=3.5)
    p.add_argument("--data-dir", default="data")
    p.set_defaults(func=cmd_radar_categories)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
