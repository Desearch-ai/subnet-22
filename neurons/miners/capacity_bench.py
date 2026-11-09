"""Measures on a real model server what the capacity test depends on. It touches neither the chain nor the task API.

    python -m neurons.miners.capacity_bench throughput --url http://127.0.0.1:8000/v1/embeddings --streams 1,5,38
    python -m neurons.miners.capacity_bench compare --url http://a:8000/v1/embeddings --other-url http://b:8000/v1/embeddings

docs/capacity-gpu-test.md says what to run, and what result passes.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import statistics
import time
from pathlib import Path

import aiohttp
import numpy as np

from desearch import capacity
from desearch.embedding import EmbeddingClient, model_named

# Same GPUs across more hotkeys must not earn more, within this share of the single hotkey's result.
NEUTRAL_WITHIN = 0.02
SEED = hashlib.sha256(b"sn22 capacity bench").digest()


def hotkey_named(index: int) -> str:
    return f"bench-hotkey-{index}"


async def stream(
    client: EmbeddingClient,
    hotkey: str,
    stop_at: float,
    in_flight: int,
    done: list[int],
    errors: list[str],
) -> None:
    """One hotkey's share of a test: its own texts, batches in flight, as many as fit before the stop."""
    turn = {"next": 0}

    async def work() -> None:
        while time.monotonic() < stop_at:
            start = turn["next"]
            turn["next"] += client.batch
            texts = [
                capacity.text_at(SEED, hotkey, index)
                for index in range(start, start + client.batch)
            ]
            try:
                await client.embed_now(texts)
            except (aiohttp.ClientError, asyncio.TimeoutError, ValueError) as exc:
                errors.append(repr(exc))
                await asyncio.sleep(0.1)
                continue
            done.append(sum(map(len, texts)))

    await asyncio.gather(*(work() for _ in range(in_flight)))


async def measure(
    client: EmbeddingClient, streams: int, seconds: float, in_flight: int
) -> dict:
    """Characters a second with this many hotkeys at once, in flight batches shared out as a miner would."""
    done: list[int] = []
    errors: list[str] = []
    each = max(1, in_flight // streams)
    # All the work started is counted, and all of its time, so more batches in flight at the end cost nothing.
    stop_at = time.monotonic() + seconds
    started = time.monotonic()
    await asyncio.gather(
        *(
            stream(client, hotkey_named(index), stop_at, each, done, errors)
            for index in range(streams)
        )
    )
    return {
        "streams": streams,
        "in_flight": each * streams,
        "seconds": seconds,
        "chars_per_second": sum(done) / max(time.monotonic() - started, 1e-6),
        "batches": len(done),
        "errors": len(errors),
    }


async def throughput(args: argparse.Namespace) -> dict:
    client = EmbeddingClient(
        args.url, model_named(args.model), args.key, in_flight=1, batch=args.batch
    )
    rows = []
    for streams in (int(part) for part in args.streams.split(",")):
        rows.append(await measure(client, streams, args.seconds, args.in_flight))
        print(json.dumps(rows[-1]), flush=True)
    await client.aclose()
    return {"check": "throughput", "rows": rows, **neutrality(rows)}


def neutrality(rows: list[dict]) -> dict:
    """More hotkeys on the same GPUs earning more than one is the failure this test exists to find."""
    base = rows[0]["chars_per_second"]
    if not base or any(row["errors"] for row in rows):
        return {"most_above_one_hotkey": None, "passes": False}
    above = max(row["chars_per_second"] / base - 1.0 for row in rows)
    return {"most_above_one_hotkey": above, "passes": above <= NEUTRAL_WITHIN}


async def compare(args: argparse.Namespace) -> dict:
    """How close two model servers' vectors are, on test texts and on your real texts: two GPUs, or a fast variant of the model."""
    model = model_named(args.model)
    one = EmbeddingClient(args.url, model, args.key, batch=args.batch)
    other = EmbeddingClient(args.other_url, model, args.key, batch=args.batch)
    sets = {
        "test_texts": [
            capacity.text_at(SEED, hotkey_named(0), index)
            for index in range(args.count)
        ]
    }
    if args.real_texts:
        lines = Path(args.real_texts).read_text().splitlines()
        sets["real_texts"] = [line for line in lines if line.strip()][: args.count]
    results = {}
    for name, texts in sets.items():
        ours = np.array(await one.embed(texts), dtype=np.float32)
        theirs = np.array(await other.embed(texts), dtype=np.float32)
        values = capacity.similarities(
            ours / np.linalg.norm(ours, axis=1, keepdims=True), theirs
        )
        results[name] = similarity_summary(values)
        print(name, json.dumps(results[name]), flush=True)
    await one.aclose()
    await other.aclose()
    return {"check": "compare", **results}


def similarity_summary(values: list[float]) -> dict:
    """How a fair check of 48 samples would fare on vectors this close."""
    below = sum(value < capacity.MATCH_SIMILARITY for value in values) / len(values)
    return {
        "texts": len(values),
        "worst": min(values),
        "median_gap": 1.0 - statistics.median(values),
        "share_below_match": below,
        "chance_a_test_passes": (1.0 - below) ** capacity.SAMPLES,
        "within_tight_rule": 1.0 - statistics.median(values)
        <= capacity.TIGHT_MEDIAN_GAP,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("check", choices=["throughput", "compare"])
    parser.add_argument("--url", required=True, help="model server; commas for several")
    parser.add_argument("--other-url", default="", help="the server to compare with")
    parser.add_argument("--key", default="")
    parser.add_argument("--model", default="desearch-4b-v1")
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--streams", default="1,5,38", help="hotkeys at once, in turn")
    parser.add_argument("--seconds", type=float, default=60.0)
    parser.add_argument("--in-flight", type=int, default=8, help="batches at once")
    parser.add_argument("--count", type=int, default=2000, help="texts to compare")
    parser.add_argument(
        "--real-texts", default="", help="a file of real texts, one a line"
    )
    parser.add_argument("--out", default="")
    args = parser.parse_args()
    if args.check == "compare" and not args.other_url:
        parser.error("compare needs --other-url")
    run = throughput if args.check == "throughput" else compare
    result = asyncio.run(run(args))
    print(json.dumps(result, indent=1))
    if args.out:
        Path(args.out).write_text(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
