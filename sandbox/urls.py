"""The URLs the sandbox serves: shards of the public dataset, or a file of your own."""

from __future__ import annotations

import random
from collections.abc import Iterator
from pathlib import Path
from urllib.parse import urlsplit

import aiohttp

DATASET = "desearch/sn22-sandbox-urls"
TREE_URL = f"https://huggingface.co/api/datasets/{DATASET}/tree/main/data"
FILE_URL = f"https://huggingface.co/datasets/{DATASET}/resolve/main/"
TIMEOUT = aiohttp.ClientTimeout(total=300)


def host_of(url: str) -> str:
    """The site a URL belongs to, as the task API groups them to spread each site over tasks."""
    host = (urlsplit(url).hostname or "").lower()
    return host.removeprefix("www.")


def read_urls(path: Path) -> list[str]:
    """A parquet file with a url column, or a text file with one URL per line."""
    if path.suffix == ".parquet":
        import pyarrow.parquet as pq

        return [
            u
            for u in pq.read_table(path, columns=["url"]).column("url").to_pylist()
            if u
        ]
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


async def shard_names(http: aiohttp.ClientSession) -> list[str]:
    async with http.get(TREE_URL) as response:
        response.raise_for_status()
        listing = await response.json()
    return sorted(e["path"] for e in listing if e["path"].endswith(".parquet"))


async def download(http: aiohttp.ClientSession, name: str, cache: Path) -> Path:
    target = cache / Path(name).name
    if target.exists():
        return target
    cache.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(".part")
    async with http.get(FILE_URL + name) as response:
        response.raise_for_status()
        with partial.open("wb") as out:
            async for chunk in response.content.iter_chunked(1 << 20):
                out.write(chunk)
    partial.rename(target)
    return target


class DatasetUrls:
    """Hands out the dataset's URLs shard by shard, in a random shard order per run."""

    def __init__(self, cache: Path, seed: int | None = None):
        self.cache = cache
        self.random = random.Random(seed)
        self.names: list[str] = []
        self.pending: Iterator[str] = iter(())

    async def take(self, count: int) -> list[str]:
        taken: list[str] = []
        async with aiohttp.ClientSession(timeout=TIMEOUT) as http:
            while len(taken) < count:
                taken += [u for _, u in zip(range(count - len(taken)), self.pending)]
                if len(taken) >= count:
                    break
                if not self.names:
                    self.names = await shard_names(http)
                    self.random.shuffle(self.names)
                name = self.names.pop()
                path = await download(http, name, self.cache)
                print(f"serving {path.name} from {DATASET}", flush=True)
                self.pending = iter(read_urls(path))
        return taken


class FileUrls:
    def __init__(self, path: Path):
        self.pending = iter(read_urls(path))

    async def take(self, count: int) -> list[str]:
        return [u for _, u in zip(range(count), self.pending)]
