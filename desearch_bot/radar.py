"""Cloudflare Radar: ranked domain lists in bulk, and per-domain categories."""

from __future__ import annotations

import asyncio
import csv
import os
import re
import time
import urllib.request
from pathlib import Path

import aiohttp

API = "https://api.cloudflare.com/client/v4"
BUCKETS = (
    200,
    500,
    1000,
    2000,
    5000,
    10000,
    20000,
    50000,
    100000,
    200000,
    500000,
    1000000,
)
# The account-wide limit is 1,200 requests per five minutes; stay under it.
RATE = 3.5
TIMEOUT = 60
BATCH = 100

# Exact names only for labels that remove a domain, so a loose match can never drop one.
EXCLUDING = {
    "adult themes": "adult",
    "pornography": "adult",
    "nudity": "adult",
    "gambling": "gambling",
}


def token() -> str:
    value = os.environ.get("CLOUDFLARE_API_TOKEN")
    if not value:
        raise RuntimeError("CLOUDFLARE_API_TOKEN is not set")
    return value


def _request(path: str, accept: str = "application/json") -> urllib.request.Request:
    return urllib.request.Request(
        f"{API}/{path}",
        headers={"Authorization": f"Bearer {token()}", "Accept": accept},
    )


def download_bucket(size: int, dest: Path) -> Path:
    """Save one ranking bucket as a single-column CSV of domains."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(
        _request(f"radar/datasets/ranking_top_{size}", "text/csv"), timeout=TIMEOUT
    ) as response:
        dest.write_bytes(response.read())
    return dest


def read_bucket(path: Path) -> set[str]:
    with open(path, encoding="utf-8", errors="replace") as handle:
        hosts = {
            row[0].strip().lower()
            for row in csv.reader(handle)
            if row and row[0].strip()
        }
    hosts.discard("domain")
    return hosts


def ranking(data_dir: Path, refresh: bool = True) -> dict[str, int]:
    """Every domain in the top million, ranked by the smallest bucket that contains it."""
    data_dir = Path(data_dir)
    ranks: dict[str, int] = {}
    for size in sorted(BUCKETS, reverse=True):
        path = data_dir / f"radar_top_{size}.csv"
        if refresh or not path.exists():
            download_bucket(size, path)
        for host in read_bucket(path):
            ranks[host] = size
    return ranks


def label(name: str) -> str:
    """Our label for a Radar category name."""
    lower = name.strip().lower()
    if lower in EXCLUDING:
        return EXCLUDING[lower]
    if "news" in lower:
        return "news"
    return re.sub(r"[^a-z0-9]+", "_", lower).strip("_")


class Categories:
    """Looks domains up on Radar at a steady rate below the account limit."""

    def __init__(self, session: aiohttp.ClientSession, rate: float = RATE):
        self.session = session
        self.interval = 1.0 / rate
        self._next = 0.0

    async def _slot(self) -> None:
        now = time.monotonic()
        start = max(self._next, now)
        self._next = start + self.interval
        if start > now:
            await asyncio.sleep(start - now)

    async def fetch(self, host: str) -> tuple[list[dict] | None, str | None]:
        """Radar's categories for a host, or None with the reason it could not be looked up."""
        await self._slot()
        try:
            async with self.session.get(
                f"{API}/radar/ranking/domain/{host}",
                headers={"Authorization": f"Bearer {token()}"},
                timeout=aiohttp.ClientTimeout(total=TIMEOUT, connect=15),
            ) as response:
                if response.status == 429:
                    await asyncio.sleep(float(response.headers.get("Retry-After", 30)))
                    return None, "rate_limited"
                if response.status != 200:
                    return None, f"http_{response.status}"
                payload = await response.json()
        except Exception as exc:
            return None, type(exc).__name__
        details = (payload.get("result") or {}).get("details_0") or {}
        return details.get("categories") or [], None


def _rows(host: str, found: list[dict]) -> list[tuple]:
    return [
        (
            host,
            "radar",
            label(item["name"]),
            item["name"],
            item.get("id"),
            None
            if item.get("superCategoryId") is None
            else str(item["superCategoryId"]),
        )
        for item in found
    ]


async def categorise(
    pool, hosts: list[str], rate: float = RATE, concurrency: int = 4, on_batch=None
) -> dict:
    """Ask Radar about each host and record the answer, including when there is none."""
    from . import categories, db

    counts = {"checked": 0, "categorised": 0, "failed": 0}
    semaphore = asyncio.Semaphore(concurrency)
    async with aiohttp.ClientSession() as session:
        radar = Categories(session, rate)

        async def one(host: str):
            async with semaphore:
                found, error = await radar.fetch(host)
                return host, found, error

        for start in range(0, len(hosts), BATCH):
            results = await asyncio.gather(
                *(one(h) for h in hosts[start : start + BATCH])
            )
            rows, checked = [], []
            for host, found, _ in results:
                if found is None:
                    counts["failed"] += 1
                    continue
                checked.append(host)
                counts["categorised"] += bool(found)
                rows.extend(_rows(host, found))
            await db.save_domain_categories(pool, rows)
            await db.save_category_checks(pool, checked, "radar")
            await db.refresh_category_rollup(pool, checked, categories.PRIORITY)
            counts["checked"] += len(checked)
            if on_batch:
                on_batch(counts)
    return counts
