import asyncio
import hashlib

import numpy as np
from aiohttp import web

from desearch.embedding import (
    INPUT_SCHEMA,
    OUTPUT_SCHEMA,
    EmbeddingModel,
    decode_vector,
    read_parquet,
    write_parquet,
)
from neurons.miners.config import Settings
from neurons.miners.embed import EmbedMiner
from tests.local_http import serving

MODEL = EmbeddingModel("tiny", "tiny/tiny", "rev", 4, "tiny")
INPUTS = [
    {
        "text_id": f"p#chunk{i}",
        "page_key": "p",
        "url": "https://a.example/p",
        "content_sha1": "c" * 40,
        "kind": "chunk",
        "index": i,
        "text": f"passage {i}",
    }
    for i in range(3)
]
GIVEN = write_parquet(INPUTS, INPUT_SCHEMA)


class Embedder:
    model = MODEL

    def __init__(self):
        self.texts: list[str] = []

    async def embed(self, texts):
        self.texts += texts
        return np.array([[i + 1.0, 0, 0, 0] for i in range(len(texts))])

    async def aclose(self):
        pass


class Api:
    hotkey = "5Miner"

    def __init__(self):
        self.posts: list[tuple[str, dict | None]] = []

    async def post(self, path, body=None):
        self.posts.append((path, body))
        return {}

    async def aclose(self):
        pass


async def run_task(model: str = "tiny", sha256: str | None = None):
    uploaded: list[bytes] = []

    async def handler(request: web.Request) -> web.Response:
        if request.method == "PUT":
            uploaded.append(await request.read())
            return web.Response()
        return web.Response(body=GIVEN)

    api, embedder = Api(), Embedder()
    async with serving(handler) as base:
        miner = EmbedMiner(Settings(), api=api, embedder=embedder)
        task = {
            "task_id": "t1",
            "model": model,
            "input": {
                "url": base + "input",
                "sha256": sha256 or hashlib.sha256(GIVEN).hexdigest(),
            },
            "upload": {
                "url": base + "upload",
                "key": "k1",
                "content_type": "application/x",
            },
        }
        try:
            await miner.process_task(task)
        finally:
            await miner.aclose()
    return api.posts, uploaded, embedder.texts


def test_every_text_gets_a_unit_vector_and_the_task_completes():
    posts, uploaded, texts = asyncio.run(run_task())

    assert texts == [row["text"] for row in INPUTS]
    rows = read_parquet(uploaded[0], OUTPUT_SCHEMA)
    assert [row["text_id"] for row in rows] == [row["text_id"] for row in INPUTS]
    assert all(decode_vector(row["vector"], 4).tolist() == [1, 0, 0, 0] for row in rows)
    assert posts == [
        (
            "/v1/tasks/t1/complete",
            {"key": "k1", "rows": 3, "ok": 3, "errors": 0, "bytes": len(uploaded[0])},
        )
    ]


def test_a_task_for_another_model_is_handed_back():
    posts, uploaded, texts = asyncio.run(run_task(model="bigger"))

    assert posts == [("/v1/tasks/t1/abandon", None)] and not uploaded and not texts


def test_an_input_that_does_not_match_its_hash_is_not_embedded():
    posts, uploaded, texts = asyncio.run(run_task(sha256="0" * 64))

    assert posts == [("/v1/tasks/t1/abandon", None)] and not uploaded and not texts
