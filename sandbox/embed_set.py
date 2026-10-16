"""Builds the sandbox's embed test set: Wikipedia articles as embed texts, each with a reference vector."""

from __future__ import annotations

import argparse
import asyncio
import itertools
import json
import os
import random
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import quote

import aiohttp
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from desearch.embedding import (
    INPUT_SCHEMA,
    LOCAL_EMBEDDINGS,
    EmbeddingClient,
    model_named,
)
from sandbox.local import ROOT

WIKIPEDIA = "wikimedia/wikipedia"
SNAPSHOT = "20231101.en"
MODEL = "desearch-4b-v1"
OUT = ROOT / "sandbox" / "cache" / "embed-set"
MIN_CHARS = 300
MAX_HEADINGS = 30
HEADING_CHARS = 100
SENTENCE_END = (".", "!", "?", ":", ";", ",")
FILE_BYTES = 45_000_000
ATTEMPTS = 3
SCHEMA = pa.schema(
    [field.with_nullable(False) for field in INPUT_SCHEMA]
    + [pa.field("reference", pa.binary(), nullable=False)]
)
CARD = """---
license: cc-by-sa-4.0
language:
- en
pretty_name: SN22 sandbox embed texts
configs:
- config_name: default
  data_files: data/*.parquet
---

# SN22 sandbox embed texts

Texts and reference vectors for testing a [Desearch Subnet 22](https://github.com/Desearch-ai/subnet-22)
embed miner locally with `python -m sandbox embed`.

{pages:,} English Wikipedia articles, made into {texts:,} texts ({page_texts:,} page texts and
{passages:,} passages) by the same code that makes mainnet embed tasks. Each row holds a text in the
columns of an embed task's input, plus `reference`: its vector from `{model}`
([`{repo}`](https://huggingface.co/{repo}) at revision `{revision}`), {dims:,} little-endian float32
values, L2-normalized. Rows are grouped by page; `manifest.json` lists the files.

The vectors were computed with vLLM 0.31.0 in float32 (`--runner pooling --dtype float32
--no-enable-chunked-prefill --max-num-batched-tokens 16384`, with `TRITON_F32_DEFAULT=ieee` so attention
runs in full float32 too), each text sent as is, with no prompt, and `truncate_prompt_tokens: 2048`.

The texts are adapted from Wikipedia, by Wikipedia contributors, through the
[`{wikipedia}`](https://huggingface.co/datasets/{wikipedia}) dataset, snapshot `{snapshot}`. Like
Wikipedia, this dataset is licensed under [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/).
"""


def blocks_of(raw: str) -> list[list[str]]:
    blocks, block = [], []
    for line in raw.splitlines():
        if line.strip():
            block.append(line)
        elif block:
            blocks.append(block)
            block = []
    return blocks + [block] if block else blocks


def squash(line: str) -> str:
    return " ".join(line.split())


def headline(line: str) -> bool:
    """A short line, not indented like a list item, that does not end like a sentence."""
    text = squash(line)
    return (
        not line[0].isspace()
        and len(text) <= HEADING_CHARS
        and not text.endswith(SENTENCE_END)
    )


def page_record(title: str, raw: str) -> dict:
    """An article as a crawled page record: squashed lines, its section headings, without the trailing category list."""
    blocks = blocks_of(raw)
    if len(blocks) > 1 and all(headline(line) for line in blocks[-1]):
        blocks.pop()
    headings = [
        squash(block[0])
        for i, block in enumerate(blocks[1:], 1)
        if headline(block[0])
        and ": " not in block[0]
        and (len(block) > 1 or i + 1 < len(blocks))
    ]
    path = quote(title.replace(" ", "_"), safe=";@$!*(),/~:")
    return {
        "url": f"https://en.wikipedia.org/wiki/{path}",
        "title": title,
        "text": "\n".join(squash(line) for block in blocks for line in block),
        "domain": "en.wikipedia.org",
        "lang": "en",
        "page_type": "article",
        "headings": headings[:MAX_HEADINGS],
    }


def sample_articles(snapshot: str, pages: int, groups: int, seed: int) -> list[dict]:
    """Random articles of at least MIN_CHARS characters, a few from each of `groups` random row groups of the snapshot."""
    from huggingface_hub import HfFileSystem

    fs = HfFileSystem()
    rng = random.Random(seed)
    shards = sorted(fs.glob(f"datasets/{WIKIPEDIA}/{snapshot}/*.parquet"))

    def group_count(shard: str) -> int:
        with fs.open(shard) as f:
            return pq.ParquetFile(f).metadata.num_row_groups

    def articles(pick: tuple[str, int]) -> list[dict]:
        with fs.open(pick[0]) as f:
            rows = pq.ParquetFile(f).read_row_group(pick[1], columns=["title", "text"])
        found = [page_record(r["title"], r["text"]) for r in rows.to_pylist()]
        return [r for r in found if len(r["text"]) >= MIN_CHARS]

    per_group = -(-pages // groups)
    records = []
    with ThreadPoolExecutor(16) as pool:
        counts = pool.map(group_count, shards)
        everywhere = [(s, g) for s, count in zip(shards, counts) for g in range(count)]
        for found in pool.map(articles, sorted(rng.sample(everywhere, groups))):
            records += rng.sample(found, min(per_group, len(found)))
    rng.shuffle(records)
    return records[:pages]


def embed_texts(records: list[dict]) -> list[dict]:
    """The planner's texts for the records, from the publisher's own Rust code."""
    lines = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records)
    done = subprocess.run(
        ["cargo", "run", "--quiet", "--example", "embed_texts"],
        cwd=ROOT / "publisher",
        input=lines.encode(),
        stdout=subprocess.PIPE,
        check=True,
    )
    return [json.loads(line) for line in done.stdout.splitlines()]


async def references(
    texts: list[str], url: str, key: str, concurrency: int
) -> np.ndarray:
    """Each text's vector from the model server, L2-normalized, as little-endian float32."""
    model = model_named(MODEL)
    client = EmbeddingClient(url, model, api_key=key, timeout=1800)
    gate = asyncio.Semaphore(concurrency)
    done = 0

    async def embed(chunk: list[str]) -> np.ndarray:
        nonlocal done
        async with gate:
            for attempt in range(ATTEMPTS):
                try:
                    vectors = await client.embed(chunk)
                    break
                except (aiohttp.ClientError, asyncio.TimeoutError):
                    if attempt + 1 == ATTEMPTS:
                        raise
        done += len(chunk)
        print(f"embedded {done}/{len(texts)}", flush=True)
        return vectors

    try:
        served = await client.served_models()
        if model.name not in served:
            raise SystemExit(f"{url} serves {served}, not {model.name}")
        chunks = [
            texts[i : i + client.batch] for i in range(0, len(texts), client.batch)
        ]
        vectors = np.concatenate(await asyncio.gather(*map(embed, chunks)))
    finally:
        await client.aclose()
    vectors = vectors.astype(np.float64)
    return (vectors / np.linalg.norm(vectors, axis=1, keepdims=True)).astype("<f4")


def write_set(out: Path, rows: list[dict], vectors: np.ndarray, snapshot: str) -> dict:
    """The data files, manifest.json and README.md, in files of whole pages under FILE_BYTES."""
    model = model_named(MODEL)
    data = out / "data"
    data.mkdir(parents=True, exist_ok=True)
    for old in data.glob("*.parquet"):
        old.unlink()
    for row, vector in zip(rows, vectors):
        row["reference"] = vector.tobytes()
    files, held, size = [], [], 0

    def write() -> None:
        name = f"data/texts-{len(files):05d}.parquet"
        table = pa.Table.from_pylist(held, schema=SCHEMA)
        pq.write_table(table, out / name, compression="zstd")
        files.append(name)

    pages = [list(page) for _, page in itertools.groupby(rows, lambda r: r["page_key"])]
    for page in pages:
        page_size = sum(len(r["text"].encode()) + len(r["reference"]) for r in page)
        if held and size + page_size > FILE_BYTES:
            write()
            held, size = [], 0
        held += page
        size += page_size
    if held:
        write()
    manifest = {
        "model": model.name,
        "repo": model.repo,
        "revision": model.revision,
        "dims": model.dims,
        "pages": len(pages),
        "texts": len(rows),
        "files": files,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    page_texts = sum(r["kind"] == "page" for r in rows)
    card = CARD.format(
        pages=len(pages),
        texts=len(rows),
        page_texts=page_texts,
        passages=len(rows) - page_texts,
        model=model.name,
        repo=model.repo,
        revision=model.revision,
        dims=model.dims,
        wikipedia=WIKIPEDIA,
        snapshot=snapshot,
    )
    (out / "README.md").write_text(card)
    return manifest


def push(out: Path, repo: str) -> None:
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(repo, repo_type="dataset", exist_ok=True)
    api.upload_folder(
        repo_id=repo,
        repo_type="dataset",
        folder_path=str(out),
        commit_message="Refresh the sandbox embed set",
        delete_patterns=["data/*.parquet"],
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m sandbox.embed_set", description=__doc__
    )
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--pages", type=int, default=3000)
    parser.add_argument(
        "--groups",
        type=int,
        default=150,
        help="random row groups of 1,000 articles to draw from",
    )
    parser.add_argument("--snapshot", default=SNAPSHOT, help=f"a {WIKIPEDIA} config")
    parser.add_argument("--seed", type=int, default=22)
    parser.add_argument(
        "--url",
        default=os.environ.get("EMBED_API_URL", LOCAL_EMBEDDINGS),
        help=f"an OpenAI-compatible embeddings endpoint serving {MODEL} in float32, started with TRITON_F32_DEFAULT=ieee",
    )
    parser.add_argument(
        "--key",
        default=os.environ.get("EMBED_API_KEY", ""),
        help="its key; EMBED_API_KEY by default",
    )
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument(
        "--push",
        metavar="REPO",
        help="upload the built --out directory to this Hugging Face dataset instead of building",
    )
    args = parser.parse_args(argv)

    if args.push:
        push(args.out, args.push)
        print(f"pushed {args.out} to {args.push}", flush=True)
        return 0
    records = sample_articles(args.snapshot, args.pages, args.groups, args.seed)
    rows = embed_texts(records)
    print(f"{len(rows)} texts from {len(records)} articles", flush=True)
    vectors = asyncio.run(
        references([r["text"] for r in rows], args.url, args.key, args.concurrency)
    )
    manifest = write_set(args.out, rows, vectors, args.snapshot)
    print(json.dumps(manifest), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
