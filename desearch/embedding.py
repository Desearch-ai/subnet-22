"""Embedding models, the files an embed task reads and writes, and a client for a model server."""

from __future__ import annotations

import asyncio
import base64
import time
from collections import deque
from dataclasses import dataclass

import aiohttp
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


@dataclass(frozen=True)
class EmbeddingModel:
    name: str
    repo: str
    revision: str
    dims: int
    max_tokens: int

    @property
    def vector_bytes(self) -> int:
        return self.dims * VECTOR_DTYPE.itemsize


MODELS = {
    "desearch-4b-v1": EmbeddingModel(
        name="desearch-4b-v1",
        repo="desearch/Desearch-Embedding-4B",
        revision="12e5c6910d49dd5534e8a77a6fa7d8a057bd534a",
        dims=2560,
        max_tokens=2048,
    ),
}
LOCAL_EMBEDDINGS = "http://127.0.0.1:8000/v1/embeddings"

INPUT_SCHEMA = pa.schema(
    [
        ("text_id", pa.string()),
        ("page_key", pa.string()),
        ("url", pa.string()),
        ("content_sha1", pa.string()),
        ("kind", pa.string()),
        ("index", pa.int32()),
        ("text", pa.large_string()),
    ]
)
VECTOR_DTYPE = np.dtype("<f2")
MAX_DECODED_BYTES = 256_000_000
BENCHED_S = 30.0
PAUSES_KEPT = 16


def model_named(name: str) -> EmbeddingModel:
    if name not in MODELS:
        raise ValueError(
            f"unknown embedding model {name!r}, expected one of {sorted(MODELS)}"
        )
    return MODELS[name]


def encode_vector(vector: np.ndarray) -> bytes:
    unit = vector / max(float(np.linalg.norm(vector)), 1e-12)
    return unit.astype(VECTOR_DTYPE).tobytes()


def encode_vectors(vectors: np.ndarray) -> bytes:
    """An embed upload: every vector at length 1, in the input's row order, with nothing around them."""
    return b"".join(encode_vector(vector) for vector in vectors)


def decode_vector(raw: bytes | None, dims: int) -> np.ndarray | None:
    if raw is None or len(raw) != dims * VECTOR_DTYPE.itemsize:
        return None
    return np.frombuffer(raw, dtype=VECTOR_DTYPE).astype(np.float32)


class EmbeddingClient:
    """OpenAI-compatible /embeddings endpoints, such as vLLM, serving the model under its name; a comma separates several GPUs."""

    def __init__(
        self,
        url: str,
        model: EmbeddingModel,
        api_key: str = "",
        batch: int = 32,
        timeout: float = 120.0,
        in_flight: int = 4,
    ):
        self.urls = [part.strip() for part in url.split(",") if part.strip()]
        self.url = self.urls[0]
        self.model = model
        self.api_key = api_key
        self.batch = batch
        self.in_flight = in_flight * len(self.urls)
        self.timeout = aiohttp.ClientTimeout(total=timeout)
        self.load = dict.fromkeys(self.urls, 0)
        self.benched: dict[str, float] = {}
        self.session: aiohttp.ClientSession | None = None
        self.free = asyncio.Event()
        self.free.set()
        self.paused_at = 0.0
        self.pauses: deque[tuple[float, float]] = deque(maxlen=PAUSES_KEPT)

    def pause(self) -> None:
        """Real tasks wait between batches, so a capacity test has the server to itself."""
        if self.free.is_set():
            self.paused_at = time.monotonic()
        self.free.clear()

    def resume(self) -> None:
        if not self.free.is_set():
            self.pauses.append((self.paused_at, time.monotonic()))
        self.free.set()

    def paused_since(self, started: float) -> float:
        """How long real tasks have been held up since a moment."""
        spans = list(self.pauses)
        if not self.free.is_set():
            spans.append((self.paused_at, time.monotonic()))
        return sum(max(0.0, end - max(begin, started)) for begin, end in spans)

    def _session(self) -> aiohttp.ClientSession:
        if self.session is None:
            headers = (
                {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
            )
            self.session = aiohttp.ClientSession(timeout=self.timeout, headers=headers)
        return self.session

    @property
    def servers(self) -> int:
        return len(self.urls)

    async def served_models(self) -> list[str]:
        """The model names every server answers to, from their /models lists."""
        served: list[str] | None = None
        for url in self.urls:
            models = url.rsplit("/embeddings", 1)[0] + "/models"
            async with self._session().get(models) as response:
                response.raise_for_status()
                answer = await response.json()
            names = [row["id"] for row in answer.get("data", [])]
            served = names if served is None else [n for n in served if n in names]
        return served or []

    def _least_busy(self) -> str:
        """The server with the fewest batches on it, so a faster GPU takes more; one that just failed only if all did."""
        now = time.monotonic()
        working = [url for url in self.urls if self.benched.get(url, 0.0) <= now]
        return min(working or self.urls, key=self.load.__getitem__)

    async def embed(self, texts: list[str]) -> np.ndarray:
        """Vectors in the texts' order; a few batches in flight keep the server busy."""
        slots = asyncio.Semaphore(self.in_flight)

        async def batch(start: int) -> list[np.ndarray]:
            async with slots:
                await self.free.wait()
                return await self.embed_now(texts[start : start + self.batch])

        parts = await asyncio.gather(
            *(batch(start) for start in range(0, len(texts), self.batch))
        )
        vectors = [vector for part in parts for vector in part]
        return np.array(vectors, dtype=np.float32).reshape(len(texts), self.model.dims)

    async def embed_now(self, texts: list[str]) -> list[np.ndarray]:
        """One batch, whether or not real tasks are paused."""
        body = {
            "model": self.model.name,
            "input": texts,
            "encoding_format": "base64",
            "truncate_prompt_tokens": self.model.max_tokens,
        }
        for tries_left in range(len(self.urls) - 1, -1, -1):
            url = self._least_busy()
            self.load[url] += 1
            try:
                async with self._session().post(url, json=body) as response:
                    response.raise_for_status()
                    answer = await response.json()
                break
            except (aiohttp.ClientError, asyncio.TimeoutError):
                # A dead server answers fast, and would otherwise always look the least busy.
                self.benched[url] = time.monotonic() + BENCHED_S
                if not tries_left:
                    raise
            finally:
                self.load[url] -= 1
        rows = sorted(answer["data"], key=lambda row: row["index"])
        vectors = [
            np.frombuffer(base64.b64decode(row["embedding"]), dtype=np.float32)
            for row in rows
        ]
        if len(vectors) != len(texts) or any(
            v.size != self.model.dims for v in vectors
        ):
            raise ValueError(
                f"the server answered {len(vectors)} vectors of {sorted({v.size for v in vectors})}"
                f" dims for {len(texts)} texts, expected {self.model.dims}"
            )
        return vectors

    async def aclose(self) -> None:
        if self.session is not None:
            await self.session.close()


def write_parquet(rows: list[dict], schema: pa.Schema) -> bytes:
    sink = pa.BufferOutputStream()
    pq.write_table(pa.Table.from_pylist(rows, schema=schema), sink, compression="zstd")
    return sink.getvalue().to_pybytes()


def read_parquet(
    body: bytes, schema: pa.Schema, max_decoded: int = MAX_DECODED_BYTES
) -> list[dict]:
    """Rows in the expected layout, or ValueError when the file is not that layout."""
    try:
        parquet = pq.ParquetFile(pa.BufferReader(body))
        decoded = sum(
            parquet.metadata.row_group(i).total_byte_size
            for i in range(parquet.metadata.num_row_groups)
        )
        if decoded > max_decoded:
            raise ValueError(f"decodes to {decoded} bytes, over {max_decoded}")
        table = parquet.read()
    except (pa.ArrowException, OSError) as exc:
        raise ValueError(f"not a parquet file: {exc}") from None
    if not set(schema.names) <= set(table.column_names):
        raise ValueError(f"expected columns {schema.names}, got {table.column_names}")
    try:
        return table.select(schema.names).cast(schema).to_pylist()
    except (pa.ArrowException, TypeError) as exc:
        raise ValueError(f"columns have the wrong types: {exc}") from None
