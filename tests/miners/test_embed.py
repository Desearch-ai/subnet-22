import asyncio
import base64
import hashlib
import time

import numpy as np
from aiohttp import web

from desearch.embedding import (
    INPUT_SCHEMA,
    MODELS,
    OUTPUT_SCHEMA,
    EmbeddingClient,
    EmbeddingModel,
    decode_vector,
    read_parquet,
    write_parquet,
)
from neurons.miners.config import Settings
from neurons.miners.embed import EmbedMiner
from tests.local_http import serving

MODEL = EmbeddingModel("tiny", "tiny/tiny", "rev", 4, 16)
INPUTS = [
    {
        "text_id": f"p{i}",
        "page_key": f"p{i}",
        "url": f"https://a.example/p{i}",
        "content_sha1": "c" * 40,
        "kind": "page",
        "index": 0,
        "text": f"page {i}",
    }
    for i in range(3)
]
GIVEN = write_parquet(INPUTS, INPUT_SCHEMA)


class Embedder:
    model = MODEL

    def __init__(self, delay: float = 0.0):
        self.delay = delay
        self.texts: list[str] = []

    async def embed(self, texts):
        await asyncio.sleep(self.delay)
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


async def run_task(
    model: str = "tiny",
    sha256: str | None = None,
    expires_in: float = 180.0,
    embedder: Embedder | None = None,
):
    uploaded: list[bytes] = []

    async def handler(request: web.Request) -> web.Response:
        if request.method == "PUT":
            uploaded.append(await request.read())
            return web.Response()
        return web.Response(body=GIVEN)

    api, embedder = Api(), embedder or Embedder()
    async with serving(handler) as base:
        miner = EmbedMiner(Settings(), api=api, embedder=embedder)
        task = {
            "task_id": "t1",
            "model": model,
            "expires_at": time.time() + expires_in,
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


def test_a_task_that_cannot_be_embedded_before_its_claim_expires_is_handed_back_early():
    slow = Embedder(delay=5.0)

    started = time.monotonic()
    posts, uploaded, _ = asyncio.run(run_task(expires_in=21.0, embedder=slow))

    assert posts == [("/v1/tasks/t1/abandon", None)] and not uploaded
    assert time.monotonic() - started < 4.0


def test_the_claim_names_the_model_the_miner_runs():
    miner = EmbedMiner(Settings(), api=Api(), embedder=Embedder())

    assert miner.claim_body() == {"kind": "embed", "count": 1, "models": ["tiny"]}


def test_the_default_model_is_the_published_desearch_model():
    model = MODELS[Settings().embed_model]

    assert (model.repo, model.revision, model.dims, model.max_tokens) == (
        "desearch/Desearch-Embedding-4B",
        "12e5c6910d49dd5534e8a77a6fa7d8a057bd534a",
        2560,
        2048,
    )


def model_server(served: list[str], dims: int = 4, asked: list | None = None):
    async def handler(request: web.Request) -> web.Response:
        if request.path.endswith("/models"):
            return web.json_response({"data": [{"id": name} for name in served]})
        body = await request.json()
        if asked is not None:
            asked.append((dict(request.headers), body))
        vector = base64.b64encode(np.ones(dims, dtype=np.float32).tobytes()).decode()
        data = [{"index": i, "embedding": vector} for i in range(len(body["input"]))]
        return web.json_response({"data": list(reversed(data))})

    return handler


def test_the_client_cuts_texts_at_the_models_token_limit_and_checks_the_vector_size():
    async def scenario():
        asked: list = []
        async with serving(model_server(["tiny"], asked=asked)) as base:
            client = EmbeddingClient(base + "v1/embeddings", MODEL, batch=2)
            try:
                vectors = await client.embed(["a", "b", "c"])
            finally:
                await client.aclose()
        async with serving(model_server(["tiny"], dims=8)) as base:
            client = EmbeddingClient(base + "v1/embeddings", MODEL)
            try:
                await client.embed(["a"])
                wrong_size = None
            except ValueError as exc:
                wrong_size = str(exc)
            finally:
                await client.aclose()
        return asked, vectors, wrong_size

    asked, vectors, wrong_size = asyncio.run(scenario())

    assert vectors.shape == (3, 4)
    assert sorted(body["input"] for _, body in asked) == [["a", "b"], ["c"]]
    assert {(body["model"], body["truncate_prompt_tokens"]) for _, body in asked} == {
        ("tiny", 16)
    }
    assert all("Authorization" not in headers for headers, _ in asked)
    assert wrong_size and "expected 4" in wrong_size


def test_the_client_keeps_a_few_batches_in_flight_and_the_vectors_in_order():
    live = {"now": 0, "most": 0}

    async def handler(request: web.Request) -> web.Response:
        body = await request.json()
        live["now"] += 1
        live["most"] = max(live["most"], live["now"])
        await asyncio.sleep(0.01 * (10 - int(body["input"][0])))
        live["now"] -= 1
        data = [
            {
                "index": i,
                "embedding": base64.b64encode(
                    np.array([float(text), 0, 0, 0], dtype=np.float32).tobytes()
                ).decode(),
            }
            for i, text in enumerate(body["input"])
        ]
        return web.json_response({"data": data})

    async def scenario():
        async with serving(handler) as base:
            client = EmbeddingClient(base + "v1/embeddings", MODEL, batch=2, in_flight=3)
            try:
                return await client.embed([str(i) for i in range(10)])
            finally:
                await client.aclose()

    vectors = asyncio.run(scenario())

    assert vectors[:, 0].tolist() == list(range(10)), "later batches answered first"
    assert live["most"] == 3


def test_another_task_is_claimed_only_if_every_held_one_still_finishes_in_time():
    miner = EmbedMiner(Settings(), api=Api(), embedder=Embedder())
    assert miner.has_room()

    miner.claimed({"task_id": "t1", "texts": 1800, "expires_at": time.time() + 300})
    assert not miner.has_room(), "one task until the pace is measured"
    miner.recent.append((1800, 20.0))
    assert miner.has_room(), "at 90 texts a second two tasks take 40 s of the claim"
    miner.recent[0] = (1800, 180.0)
    assert not miner.has_room(), "at 10 texts a second they would outlast it"


def test_held_tasks_are_embedded_one_after_another():
    class Counting(Embedder):
        def __init__(self):
            super().__init__(delay=0.05)
            self.live = self.most = 0

        async def embed(self, texts):
            self.live += 1
            self.most = max(self.most, self.live)
            try:
                return await super().embed(texts)
            finally:
                self.live -= 1

    async def handler(request: web.Request) -> web.Response:
        if request.method == "PUT":
            await request.read()
            return web.Response()
        return web.Response(body=GIVEN)

    async def scenario():
        embedder, api = Counting(), Api()
        async with serving(handler) as base:
            miner = EmbedMiner(Settings(), api=api, embedder=embedder)
            tasks = [
                {
                    "task_id": f"t{i}",
                    "model": "tiny",
                    "texts": 3,
                    "expires_at": time.time() + 180,
                    "input": {"url": base + "input", "sha256": hashlib.sha256(GIVEN).hexdigest()},
                    "upload": {"url": base + "upload", "key": f"k{i}", "content_type": "application/x"},
                }
                for i in range(2)
            ]
            for task in tasks:
                miner.claimed(task)
            try:
                await asyncio.gather(*(miner.process_task(task) for task in tasks))
            finally:
                await miner.aclose()
        return embedder, api, miner

    embedder, api, miner = asyncio.run(scenario())

    assert embedder.most == 1
    assert sorted(path for path, _ in api.posts) == ["/v1/tasks/t0/complete", "/v1/tasks/t1/complete"]
    assert not miner.ahead and len(miner.recent) == 2


def test_the_miner_starts_only_once_its_server_serves_the_model():
    async def scenario(served):
        async with serving(model_server(served)) as base:
            settings = Settings(embed_api_url=base + "v1/embeddings")
            client = EmbeddingClient(settings.embed_api_url, MODEL)
            miner = EmbedMiner(settings, api=Api(), embedder=client)
            try:
                return await miner.server_ready()
            finally:
                await miner.aclose()

    assert asyncio.run(scenario(["tiny"])) is True
    assert asyncio.run(scenario(["Qwen/Qwen3-Embedding-4B"])) is False
