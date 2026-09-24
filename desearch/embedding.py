"""Embedding models, the files an embed task reads and writes, and a client for hosted models."""

from __future__ import annotations

import base64
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
    # The same weights on OpenAI-compatible hosts, for anyone who does not run them locally.
    hosted: str


MODELS = {
    "qwen3-embedding-8b": EmbeddingModel(
        name="qwen3-embedding-8b",
        repo="Qwen/Qwen3-Embedding-8B",
        revision="1d8ad4ca9b3dd8059ad90a75d4983776a23d44af",
        dims=4096,
        hosted="qwen/qwen3-embedding-8b",
    ),
}
TEXT_KINDS = ("head", "full", "chunk")
OPENROUTER_EMBEDDINGS = "https://openrouter.ai/api/v1/embeddings"
LOCAL_EMBEDDINGS = "http://127.0.0.1:8000/v1/embeddings"

# Texts are sized by the publisher to fit the model, so nobody truncates them.
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
OUTPUT_SCHEMA = pa.schema([("text_id", pa.string()), ("vector", pa.binary())])
VECTOR_DTYPE = np.dtype("<f2")
MAX_DECODED_BYTES = 256_000_000


def model_named(name: str) -> EmbeddingModel:
    if name not in MODELS:
        raise ValueError(f"unknown embedding model {name!r}")
    return MODELS[name]


def encode_vector(vector: np.ndarray) -> bytes:
    unit = vector / max(float(np.linalg.norm(vector)), 1e-12)
    return unit.astype(VECTOR_DTYPE).tobytes()


def decode_vector(raw: bytes | None, dims: int) -> np.ndarray | None:
    if raw is None or len(raw) != dims * VECTOR_DTYPE.itemsize:
        return None
    return np.frombuffer(raw, dtype=VECTOR_DTYPE).astype(np.float32)


class EmbeddingClient:
    """An OpenAI-compatible /embeddings endpoint: a miner's own GPU server, or a validator's hosted one."""

    def __init__(
        self,
        url: str,
        api_key: str,
        model: EmbeddingModel,
        providers: tuple[str, ...] = (),
        served_as: str = "",
        batch: int = 32,
        timeout: float = 120.0,
    ):
        self.url = url
        self.api_key = api_key
        self.model = model
        self.providers = providers
        self.served_as = served_as or model.hosted
        self.batch = batch
        self.timeout = aiohttp.ClientTimeout(total=timeout)
        self.session: aiohttp.ClientSession | None = None

    async def embed(self, texts: list[str]) -> np.ndarray:
        if self.session is None:
            self.session = aiohttp.ClientSession(timeout=self.timeout)
        vectors = []
        for start in range(0, len(texts), self.batch):
            vectors += await self._embed(texts[start : start + self.batch])
        return np.array(vectors, dtype=np.float32).reshape(len(texts), self.model.dims)

    async def _embed(self, texts: list[str]) -> list[np.ndarray]:
        body = {"model": self.served_as, "input": texts, "encoding_format": "base64"}
        if self.providers:
            body["provider"] = {"order": list(self.providers), "allow_fallbacks": False}
        async with self.session.post(
            self.url, json=body, headers={"Authorization": f"Bearer {self.api_key}"}
        ) as response:
            response.raise_for_status()
            answer = await response.json()
        rows = sorted(answer["data"], key=lambda row: row["index"])
        return [
            np.frombuffer(base64.b64decode(row["embedding"]), dtype=np.float32)
            for row in rows
        ]

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
