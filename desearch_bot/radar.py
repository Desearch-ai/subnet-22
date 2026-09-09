"""Cloudflare Radar: a ranked domain list in bulk, and per-domain categories one at a time.

The ranking buckets are nested, so the smallest bucket a domain appears in is its rank. Categories
have no bulk endpoint and the Cloudflare API allows 1,200 requests per five minutes across the
account, so they are fetched at a fixed rate by a job of their own.
"""

from __future__ import annotations

import asyncio
import csv
import json
import os
import time
import urllib.request
from pathlib import Path

API = "https://api.cloudflare.com/client/v4"
BUCKETS = (200, 500, 1000, 2000, 5000, 10000, 20000, 50000, 100000, 200000, 500000, 1000000)

# The documented limit is 1,200 requests per five minutes; stay under it.
RATE = 3.5
TIMEOUT = 60


def token() -> str:
    value = os.environ.get("CLOUDFLARE_API_TOKEN")
    if not value:
        raise RuntimeError("CLOUDFLARE_API_TOKEN is not set")
    return value


def _request(path: str, accept: str = "application/json"):
    return urllib.request.Request(
        f"{API}/{path}",
        headers={"Authorization": f"Bearer {token()}", "Accept": accept},
    )


def download_bucket(size: int, dest: Path) -> Path:
    """One bucket as CSV: a single column of domains, unordered within the bucket."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(
        _request(f"radar/datasets/ranking_top_{size}", "text/csv"), timeout=TIMEOUT
    ) as response:
        dest.write_bytes(response.read())
    return dest


def read_bucket(path: Path) -> set[str]:
    with open(path, encoding="utf-8", errors="replace") as handle:
        reader = csv.reader(handle)
        header = next(reader, None)
        if header and header[0].strip().lower() != "domain":
            handle.seek(0)
            reader = csv.reader(handle)
        return {row[0].strip().lower() for row in reader if row and row[0].strip()}


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


class Categories:
    """Per-domain categories, paced to the account-wide request limit."""

    def __init__(self, session, rate: float = RATE):
        self.session = session
        self.interval = 1.0 / rate
        self._next = 0.0

    async def _wait(self) -> None:
        remaining = self._next - time.monotonic()
        if remaining > 0:
            await asyncio.sleep(remaining)
        self._next = time.monotonic() + self.interval

    async def fetch(self, host: str) -> tuple[str, list[dict] | None, str | None]:
        """Returns (host, categories, error). An empty list means Radar knows the domain but
        has no category for it; None means it could not be looked up."""
        await self._wait()
        url = f"{API}/radar/ranking/domain/{host}"
        headers = {"Authorization": f"Bearer {token()}"}
        try:
            async with self.session.get(url, headers=headers, timeout=self.session_timeout()) as r:
                if r.status == 429:
                    await asyncio.sleep(5)
                    return host, None, "rate_limited"
                if r.status == 404:
                    return host, None, "not_ranked"
                if r.status != 200:
                    return host, None, f"http_{r.status}"
                payload = await r.json()
        except Exception as exc:
            return host, None, type(exc).__name__
        result = (payload.get("result") or {}).get("details_0") or payload.get("result") or {}
        return host, result.get("categories") or [], None

    def session_timeout(self):
        import aiohttp

        return aiohttp.ClientTimeout(total=TIMEOUT, connect=15)
