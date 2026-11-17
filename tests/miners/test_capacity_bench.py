import argparse
import asyncio
import base64

import numpy as np
from aiohttp import web

from desearch import capacity
from desearch.embedding import EmbeddingClient, EmbeddingModel
from neurons.miners import capacity_bench as bench
from tests.local_http import serving

DIMS = 8
MODEL = EmbeddingModel("tiny", "tiny/tiny", "rev", DIMS, 16)


def gpu_server(chars_per_second: float, noise: float = 0.0):
    """A model server with one GPU: requests wait their turn, whichever hotkey sent them."""
    lock = asyncio.Lock()

    async def handler(request: web.Request) -> web.Response:
        body = await request.json()
        async with lock:
            await asyncio.sleep(sum(map(len, body["input"])) / chars_per_second)
        rows = []
        for index, text in enumerate(body["input"]):
            seed = abs(hash(text)) % 2**32
            vector = np.random.default_rng(seed).normal(size=DIMS).astype(np.float32)
            vector[0] += noise
            rows.append(
                {
                    "index": index,
                    "embedding": base64.b64encode(vector.tobytes()).decode(),
                }
            )
        return web.json_response({"data": rows})

    return handler


async def test_more_hotkeys_on_one_gpu_measure_about_the_same():
    async with serving(gpu_server(2_000_000)) as url:
        client = EmbeddingClient(url + "/v1/embeddings", MODEL, batch=8, in_flight=1)
        one = await bench.measure(client, 1, 0.6, 8)
        many = await bench.measure(client, 8, 0.6, 8)
        await client.aclose()
    assert one["errors"] == many["errors"] == 0
    assert 0.7 < many["chars_per_second"] / one["chars_per_second"] < 1.3


def row(chars_per_second: float, errors: int = 0) -> dict:
    return {"chars_per_second": chars_per_second, "errors": errors}


def test_the_verdict_fails_when_more_hotkeys_earn_more():
    assert bench.neutrality([row(100.0), row(101.0)])["passes"]
    verdict = bench.neutrality([row(100.0), row(101.0), row(120.0)])
    assert not verdict["passes"] and verdict["most_above_one_hotkey"] > 0.19


def test_the_verdict_fails_on_a_dead_server_or_any_errors():
    assert not bench.neutrality([row(0.0), row(0.0)])["passes"]
    assert not bench.neutrality([row(100.0), row(100.0, errors=1)])["passes"]


async def test_many_hotkeys_on_a_neutral_gpu_do_not_read_low():
    async with serving(gpu_server(2_000_000)) as url:
        client = EmbeddingClient(url + "/v1/embeddings", MODEL, batch=8, in_flight=1)
        one = await bench.measure(client, 1, 0.5, 8)
        many = await bench.measure(client, 38, 0.5, 8)
        await client.aclose()
    assert many["chars_per_second"] / one["chars_per_second"] > 0.9


async def test_compare_finds_identical_servers_identical(monkeypatch, tmp_path):
    monkeypatch.setitem(bench.model_named.__globals__["MODELS"], "tiny", MODEL)
    real = tmp_path / "real.txt"
    real.write_text("a real page\nanother real page\n\n")
    async with serving(gpu_server(1e9)) as one, serving(gpu_server(1e9)) as two:
        args = argparse.Namespace(
            url=one + "/v1/embeddings",
            other_url=two + "/v1/embeddings",
            model="tiny",
            key="",
            batch=4,
            count=8,
            real_texts=str(real),
        )
        result = await bench.compare(args)
    assert result["test_texts"]["worst"] > 0.9999
    assert result["test_texts"]["within_tight_rule"]
    assert result["real_texts"]["texts"] == 2


def test_a_summary_says_how_a_real_check_would_fare():
    honest = bench.similarity_summary([0.999999] * 100)
    assert honest["share_below_match"] == 0.0 and honest["chance_a_test_passes"] == 1.0
    assert honest["within_tight_rule"]
    rough = bench.similarity_summary([0.9995] * 99 + [0.99])
    assert not rough["within_tight_rule"]
    assert rough["share_below_match"] == 0.01
    assert 0.6 < rough["chance_a_test_passes"] < 0.63
    assert capacity.SAMPLES == 48
