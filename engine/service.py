"""Search API over the unified index, served to desearch-public-api `/search`.

uvicorn engine.service:app --host 0.0.0.0 --port 8090
"""

from __future__ import annotations

import asyncio
import base64
import hmac
import json
import logging
import os
import threading
from contextlib import asynccontextmanager
from datetime import date
from pathlib import Path
from typing import Literal

import aiohttp
import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from .build import LIVE, OUT, canon, ready_segments
from .chunking import para_chunks
from .search import Index, Unified, date_window, unit

INSTRUCT = "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery:"
EMBED_MODEL = os.environ.get("UNIFIED_EMBED_MODEL", "qwen/qwen3-embedding-8b")
EMBED_URL = "https://openrouter.ai/api/v1/embeddings"
EMBED_TIMEOUT = aiohttp.ClientTimeout(total=30)
# DeepInfra answers in ~0.5s where Nebius takes 10-20s; SiliconFlow serves fp8, which
# would not match the vectors the index was built from.
EMBED_PROVIDER = {"order": ["DeepInfra", "Nebius"], "ignore": ["SiliconFlow"]}
ACCESS_KEY_PATH = Path(
    os.environ.get("UNIFIED_ACCESS_KEY_FILE", "/opt/tc/.index_access_key")
)
LOOPBACK = {"127.0.0.1", "::1"}

# Ranking weights per mode, as benchmarked in desearch-search-evals.
PROFILES = {
    "fast": {"head": 2, "bm_chunk": 2},
    "balanced": {"full": 1, "head": 2, "chunk": 2, "bm_chunk": 2, "bm_doc": 4},
}
DATE_BOOST = 4.0
RELOAD_S = 30.0
PASSAGE_CHARS = 1200
UNDATED = date(1970, 1, 1)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Load the index before the first request, which otherwise waits ~40s for it."""
    engine()
    query_vectors()
    _state["http"] = aiohttp.ClientSession(timeout=EMBED_TIMEOUT)
    refreshing = asyncio.create_task(refresh_forever())
    try:
        yield
    finally:
        refreshing.cancel()
        await _state.pop("http").close()


app = FastAPI(title="desearch-index", lifespan=lifespan)
_state: dict = {}
_lock = threading.Lock()
log = logging.getLogger("engine")


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    mode: Literal["fast", "balanced"] = "balanced"
    count: int = Field(10, ge=1, le=50)
    start_date: date | None = None
    end_date: date | None = None
    highlights: bool = True
    page_text: bool = False


def engine() -> Index:
    """Load the index once; refresh_forever swaps in new segments as they are synced."""
    with _lock:
        if "engine" not in _state:
            _state["engine"] = load_index()
        return _state["engine"]


def load_index(previous: Index | None = None) -> Index:
    """The built index and every ready segment not folded into it, reusing what is loaded."""
    merged_path = OUT / "merged.json"
    merged = set(json.loads(merged_path.read_text())) if merged_path.exists() else set()
    roots = [OUT] if (OUT / "docs.jsonl").exists() else []
    roots += [seg for seg in ready_segments(LIVE) if seg.name not in merged]
    loaded = {seg.root: seg for seg in previous.segments} if previous else {}
    return Index(
        [
            loaded[root]
            if root in loaded and loaded[root].is_current()
            else Unified(root)
            for root in roots
        ]
    )


async def refresh_forever() -> None:
    while True:
        await asyncio.sleep(RELOAD_S)
        try:
            await asyncio.to_thread(refresh)
        except Exception:
            log.exception("could not load new segments")


def refresh() -> bool:
    """Swaps in the index with any segment synced or rebuilt since it was loaded."""
    current = engine()
    fresh = load_index(current)
    if [s.built for s in fresh.segments] == [s.built for s in current.segments]:
        return False
    _state["engine"] = fresh
    log.info("index now %d pages in %d parts", len(fresh.meta), len(fresh.segments))
    return True


def query_vectors() -> dict[str, np.ndarray]:
    """Question vectors saved by the benchmark, so a repeated query costs no embedding call."""
    with _lock:
        if "vectors" not in _state:
            path = OUT / "qvec.npz"
            if path.exists():
                cached = np.load(path, allow_pickle=True)
                _state["vectors"] = dict(zip(cached["texts"].tolist(), cached["vecs"]))
            else:
                _state["vectors"] = {}
        return _state["vectors"]


def access_key() -> str:
    with _lock:
        if "key" not in _state:
            _state["key"] = (
                ACCESS_KEY_PATH.read_text().strip() if ACCESS_KEY_PATH.exists() else ""
            )
        return _state["key"]


async def embed(query: str) -> np.ndarray:
    async with _state["http"].post(
        EMBED_URL,
        headers={"Authorization": f"Bearer {os.environ.get('OPENROUTER_API_KEY', '')}"},
        json={
            "model": EMBED_MODEL,
            "input": [INSTRUCT + query],
            "provider": EMBED_PROVIDER,
            "encoding_format": "base64",
        },
    ) as response:
        response.raise_for_status()
        answer = await response.json()
    raw = base64.b64decode(answer["data"][0]["embedding"])
    return unit(np.frombuffer(raw, dtype=np.float32))


def passage(
    eng: Index, doc: int, vector: np.ndarray, target: int = PASSAGE_CHARS
) -> str:
    """The chunk closest to the query, widened with neighbours that fit, so a result reads."""
    part, doc = eng.locate(doc)
    start = int(np.searchsorted(part.owner, doc, side="left"))
    end = int(np.searchsorted(part.owner, doc, side="right"))
    chunks = para_chunks(part.text(doc))
    if end <= start or not chunks:
        return part.text(doc)[:target]

    best = int(np.argmax(part._rescore_chunks(np.arange(start, end), vector)))
    lo, hi, size = best, best + 1, len(chunks[best])
    grew = True
    while size < target and grew:
        grew = False
        if hi < len(chunks) and size + len(chunks[hi]) <= target + 400:
            size, hi, grew = size + len(chunks[hi]), hi + 1, True
        if size < target and lo > 0 and size + len(chunks[lo - 1]) <= target + 400:
            size, lo, grew = size + len(chunks[lo - 1]), lo - 1, True
    return "\n".join(chunks[lo:hi])


def passage_similarity(arms: dict) -> float:
    """Cosine of the query against the closest passage in the corpus, the caller's confidence signal."""
    scores = arms.get("chunk", ((), ()))[1]
    return float(max(scores)) if len(scores) else 0.0


def within_dates(
    eng: Unified, ranked: np.ndarray, start: date | None, end: date | None
):
    """Drop results outside the window; a document with no known date never matches one."""
    if not start and not end:
        return ranked

    published = eng.pub[ranked]
    keep = published != UNDATED
    if start:
        keep &= published >= start
    if end:
        keep &= published <= end
    return ranked[keep]


@app.middleware("http")
async def require_access_key(request: Request, call_next):
    if request.client and request.client.host in LOOPBACK:
        return await call_next(request)
    if not hmac.compare_digest(request.headers.get("Access-Key", ""), access_key()):
        return JSONResponse({"detail": "Unauthorized"}, status_code=401)
    return await call_next(request)


class DocumentRequest(BaseModel):
    url: str = Field(min_length=1, max_length=4000)


@app.post("/v1/document")
def document(body: DocumentRequest):
    """Full text of one indexed page, so an agent reads the corpus rather than the live web."""
    eng = engine()
    doc = eng.key_ix.get(canon(body.url))
    if doc is None:
        raise HTTPException(404, "Document not in the index")
    meta = eng.meta[doc]
    return {"url": meta["url"], "title": meta["title"], "text": eng.text(doc)}


@app.get("/healthz")
def healthz():
    return {"ok": True}


@app.post("/v1/search")
async def search(body: SearchRequest):
    query = body.query.strip()
    vector = query_vectors().get(query)
    if vector is None:
        try:
            vector = await embed(query)
        except Exception as exc:
            raise HTTPException(502, "Query embedding unavailable") from exc
    # Ranking is numpy work; off the event loop so one query does not stall the rest.
    return await asyncio.to_thread(ranked_results, engine(), body, query, vector)


def ranked_results(
    eng: Index, body: SearchRequest, query: str, vector: np.ndarray
) -> dict:
    arms, _ = eng.arms(query, vector)
    ranked, _ = eng.fuse(
        arms, PROFILES[body.mode], DATE_BOOST, date_window(query, years=True)
    )
    ranked = within_dates(eng, ranked, body.start_date, body.end_date)
    confidence = {
        "top_score": passage_similarity(arms),
        "term_coverage": eng.term_coverage(query),
    }

    results = []
    for doc in ranked[: body.count].tolist():
        meta = eng.meta[doc]
        result = {
            "url": meta["url"],
            "title": meta["title"],
            "published_date": meta["published"] or None,
        }
        if body.highlights:
            result["highlights"] = [passage(eng, doc, vector)]
        if body.page_text:
            result["page_text"] = eng.text(doc)
        results.append(result)
    return {"results": results, "confidence": confidence}
