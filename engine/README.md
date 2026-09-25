# engine

Flat-file index and search API behind `POST /search` in desearch-public-api. Everything is held
in one process, there is no reranker, and a query is answered in about 100 ms.

## How it ranks

Five arms over the same corpus:

| Arm | What it matches |
|---|---|
| `head` | Embedding of the title and opening text |
| `full` | Embedding of the whole document |
| `chunk` | Embedding of each passage |
| `bm_chunk` | BM25 keyword postings over passages |
| `bm_doc` | BM25 over the whole page, each term counted once at its best passage |

Dense arms take binary-256 candidates from the vector prefix, then rescore the candidates with
the full vectors. Arm scores are turned into z-scores and added with per-mode weights, and a
document whose publication date falls inside the window the question implies (`September 2026`,
`Q3 2025`) gets a fixed boost.

Modes, as benchmarked in desearch-search-evals:

| Mode | Weights |
|---|---|
| `fast` | head 2, bm_chunk 2 |
| `balanced` | full 1, head 2, chunk 2, bm_chunk 2, bm_doc 4 |

## Layout

- `build.py` — builds `$UNIFIED_DIR` from saved document vectors and text; re-embeds nothing.
- `search.py` — loads those artifacts and serves candidates, fusion and document text.
- `service.py` — the HTTP API: `POST /v1/search`, `POST /v1/document` and `GET /healthz`.
- `chunking.py` — how a page becomes passages plus the `head` and `full` texts.
- `sync.py` — turns vectors verified on the subnet into live segments.

Artifacts (`$UNIFIED_DIR`, default `/opt/unified`): `docs.jsonl`, `head.npy`, `full.npy`,
`chunk_*.npy`, `bm_*.npy`, `files.json`, and `qvec.npz`, a cache of benchmark question vectors
so a repeated query needs no embedding call.

## Running

From the repository root, with `engine/requirements.txt` installed:

```bash
UNIFIED_DIR=/opt/unified OPENROUTER_API_KEY=... \
  uvicorn engine.service:app --host 0.0.0.0 --port 8090
```

The index stays resident, so the process needs roughly 7 GB for the current corpus and takes
about 30 seconds to start.

| Variable | Default | Meaning |
|---|---|---|
| `UNIFIED_DIR` | `/opt/unified` | Artifact directory |
| `UNIFIED_ACCESS_KEY_FILE` | `/opt/tc/.index_access_key` | File holding the shared access key |
| `UNIFIED_EMBED_MODEL` | `qwen/qwen3-embedding-8b` | Query embedding model on OpenRouter |
| `OPENROUTER_API_KEY` | — | Required for queries that are not in `qvec.npz` |
| `UNIFIED_CAND_DOC`, `UNIFIED_CAND_CHUNK` | 200, 1000 | Candidates each dense arm rescores |

Requests from outside the machine must send `Access-Key`; loopback callers (an SSH tunnel) are
allowed through so the dashboard keeps working.

## Live updates

Pages crawled and embedded on the subnet reach the index without a rebuild. `python -m engine.sync`
polls the pages bucket for new `vectors/model=<name>/` files (the last three days, each file once,
tracked in `sync.db`), reads each page's record, and keeps a page only while the record still holds
the text its vectors were made from; a page changed since then waits for its own vectors. Each pass
writes the vectors to a store shard in `$UNIFIED_STORE` and builds a segment in `$UNIFIED_LIVE` in
the same layout as the main index, with keyword weights on the main index's scale so the two rank
together. A segment is marked `READY` only once complete.

The service loads the main index plus every ready segment and checks for new ones every 30 seconds,
without reloading what it already holds. A page in several parts is served from the newest, so a
recrawl replaces the older copy.

To fold segments into the main index, rebuild with the `live` source, e.g.
`python -m engine.build live,bench,techcrunch,news`: newer copies win, the segments folded in are
listed in `merged.json` and skipped by the service, and their directories can then be deleted (the
store shards stay; the rebuilt index reads them). Build into a new `UNIFIED_DIR` and move it into
place when it is done, never over the directory being served: the next check sees the new index and
loads it.

| Variable | Default | Meaning |
|---|---|---|
| `UNIFIED_LIVE` | `$UNIFIED_DIR` + `_live` | Segments and `sync.db` |
| `UNIFIED_STORE` | `$UNIFIED_DIR` + `_store` | Vector shards the segments and rebuilds read |
| `ENGINE_EMBED_MODEL` | `qwen3-embedding-8b` | Which model's vectors to sync; queries must use the same model |
| `CF_R2_ENDPOINT`, `CF_R2_ACCESS_KEY_ID`, `CF_R2_SECRET_ACCESS_KEY` | — | Read access to the pages bucket |
| `CF_R2_PAGES_BUCKET` | `desearch-pages` | |

## Request

```json
{"query": "...", "mode": "balanced", "count": 10,
 "start_date": "2026-09-01", "end_date": "2026-09-15",
 "highlights": true, "page_text": false}
```

Each result carries `url`, `title`, `published_date`, and the ~1200-character passage closest to
the query when `highlights` is set. A date filter drops documents whose publication date is
unknown.

## Tests

```bash
pip install -r engine/requirements.txt pytest
cd engine && pytest
```
