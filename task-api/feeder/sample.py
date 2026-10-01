"""Exports listed page URLs from desearch-bot's stores as the miner sandbox dataset."""

from __future__ import annotations

import argparse
import heapq
import random
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

from feeder.bot_stores import (
    BUCKETS_DIR,
    RECORD,
    host_prefix,
    open_store,
    parse_listed_page,
    range_end,
    store_index,
)

SECONDARY_DIR = "/var/lib/desearch-bot/tmp/sandbox-sample"
REPO = "desearch/sn22-sandbox-urls"
PER_DOMAIN = 300
# Bounds the read on huge sites; the newest pages are picked within this window.
MAX_SCAN = 100_000
SHARD_ROWS = 100_000
READAHEAD = 2 << 20
MAX_URL_CHARS = 512
IO_PRESSURE = Path("/proc/pressure/io")
IO_PRESSURE_PAUSE = 30.0
NOT_A_PAGE = re.compile(
    r"\.(pdf|jpe?g|png|gif|webp|avif|svg|ico|css|js|json|xml|rss|atom|txt|csv|zip|gz|"
    r"mp[34]|m4a|mov|avi|webm|docx?|xlsx?|pptx?|epub)(\?|#|$)",
    re.I,
)


# A localized copy of an English site, such as /de/ or /pt-br/.
OTHER_LANGUAGE = re.compile(
    r"^https?://[^/]+/(ar|bg|cs|da|de|el|es|et|fa|fi|fr|he|hi|hr|hu|id|it|ja|ko|lt|lv|ms|nb|nl|"
    r"no|pl|pt|ro|ru|sk|sl|sr|sv|th|tr|vi|zh)([-_][a-z]{2,4})?(/|$)",
    re.I,
)


def is_page(url: str) -> bool:
    return (
        len(url) <= MAX_URL_CHARS
        and not NOT_A_PAGE.search(url)
        and not OTHER_LANGUAGE.search(url)
    )


def newest_pages(store, host: str, keep: int, max_scan: int) -> list[str]:
    """The newest listed pages of one site, reading at most max_scan of its records."""
    from rocksdict import ReadOptions

    prefix = host_prefix(host)
    options = ReadOptions()
    options.set_iterate_upper_bound(range_end(prefix))
    options.fill_cache(False)
    options.set_readahead_size(READAHEAD)
    best: list[tuple] = []
    for scanned, (key, raw) in enumerate(
        store.items(from_key=prefix, read_opt=options)
    ):
        if scanned >= max_scan or not key.startswith(prefix):
            break
        _, lastmod, first_seen, *_ = RECORD.unpack(raw)
        entry = (lastmod, first_seen, bytes(key), bytes(raw))
        if len(best) < keep * 2:
            heapq.heappush(best, entry)
        elif entry[:2] > best[0][:2]:
            heapq.heapreplace(best, entry)
    pages = [
        parse_listed_page(key, raw).url for *_, key, raw in sorted(best, reverse=True)
    ]
    return [url for url in pages if is_page(url)][:keep]


def io_busy() -> bool:
    """True while the volume is saturated, so the export yields to the crawler."""
    try:
        full = IO_PRESSURE.read_text().splitlines()[1]
    except (OSError, IndexError):
        return False
    return float(full.split()[1].split("=")[1]) > IO_PRESSURE_PAUSE


def collect(hosts: list[str], root: str, per_domain: int, max_scan: int) -> list[str]:
    by_bucket: dict[int, list[str]] = defaultdict(list)
    for host in hosts:
        by_bucket[store_index(host)].append(host)
    urls: list[str] = []
    done = 0
    for bucket, owned in sorted(by_bucket.items()):
        store = open_store(root, bucket, SECONDARY_DIR)
        try:
            for host in owned:
                while io_busy():
                    time.sleep(5)
                urls += newest_pages(store, host, per_domain, max_scan)
                done += 1
        finally:
            store.close()
        print(
            f"bucket {bucket:03d}: {done}/{len(hosts)} sites, {len(urls)} urls",
            flush=True,
        )
    return urls


def write_shards(urls: list[str], out: Path, seed: int) -> list[Path]:
    import pyarrow as pa
    import pyarrow.parquet as pq

    unique = list(dict.fromkeys(urls))
    random.Random(seed).shuffle(unique)
    data = out / "data"
    data.mkdir(parents=True, exist_ok=True)
    shards = []
    for index, start in enumerate(range(0, len(unique), SHARD_ROWS)):
        path = data / f"urls-{index:05d}.parquet"
        table = pa.table({"url": unique[start : start + SHARD_ROWS]})
        pq.write_table(table, path, compression="zstd")
        shards.append(path)
    return shards


def push(out: Path, repo: str, token: str, card: Path) -> None:
    from huggingface_hub import HfApi

    (out / "README.md").write_text(card.read_text())
    HfApi(token=token).upload_folder(
        repo_id=repo,
        repo_type="dataset",
        folder_path=str(out),
        commit_message="Refresh the sandbox URLs",
        delete_patterns=["data/*.parquet"],
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m feeder.sample", description=__doc__
    )
    parser.add_argument("--hosts", required=True, help="one site per line")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--buckets", default=BUCKETS_DIR)
    parser.add_argument("--per-domain", type=int, default=PER_DOMAIN)
    parser.add_argument("--max-scan", type=int, default=MAX_SCAN)
    parser.add_argument("--seed", type=int, default=22)
    parser.add_argument("--push", action="store_true", help="upload to --repo")
    parser.add_argument("--repo", default=REPO)
    parser.add_argument(
        "--card", type=Path, default=Path(__file__).with_name("sample_card.md")
    )
    args = parser.parse_args(argv)

    hosts = [h.strip() for h in Path(args.hosts).read_text().splitlines() if h.strip()]
    urls = collect(hosts, args.buckets, args.per_domain, args.max_scan)
    shards = write_shards(urls, args.out, args.seed)
    print(
        f"{len(urls)} urls from {len(hosts)} sites in {len(shards)} shards", flush=True
    )
    if args.push:
        import os

        push(args.out, args.repo, os.environ["HF_TOKEN"], args.card)
        print(f"pushed to {args.repo}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
