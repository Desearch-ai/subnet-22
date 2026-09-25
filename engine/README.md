# Engine

The search index and API built from the pages and vectors the subnet verifies.

## How search works

Every page is indexed three ways, cut by [`chunking.py`](./chunking.py): its head (title and opening
text), its full text, and each of its passages. Each has a vector from the embedding model, and the
passages are also indexed by keyword (BM25).

A query is embedded with the same model and matched five ways:

| Arm | What it matches |
| --- | --- |
| `head` | the query's vector against page heads |
| `full` | the query's vector against whole pages |
| `chunk` | the query's vector against passages |
| `bm_chunk` | the query's keywords against passages |
| `bm_doc` | the query's keywords against whole pages |

Each arm's scores are normalized and added with per-mode weights (`fast` uses fewer arms than
`balanced`), and pages published inside a date the query names (`September 2026`, `Q3 2025`) are
lifted. Each result carries the passage closest to the query.

## How new pages arrive

The main index is built in one pass by [`build.py`](./build.py). Pages verified after that arrive
through [`sync.py`](./sync.py): it watches storage for newly published vectors, checks each page is
still at the version its vectors were made from, and adds them to the index as a small live segment.
The service picks up new segments while it runs, and a page's newest version replaces older ones.
From time to time a rebuild folds the segments into the main index.

## Running

With [`requirements.txt`](./requirements.txt) installed, from the repository root:

```bash
uvicorn engine.service:app --host 0.0.0.0 --port 8090   # search API
python -m engine.sync                                  # adds newly verified pages
python -m engine.build live,news                       # rebuilds the main index from its sources
```

Build a new index into a fresh directory and move it into place when it is done; the running service
loads it on its next check.

| Variable | Default | |
| --- | --- | --- |
| `UNIFIED_DIR` | `/opt/unified` | the main index |
| `UNIFIED_LIVE` | `$UNIFIED_DIR` + `_live` | live segments |
| `UNIFIED_STORE` | `$UNIFIED_DIR` + `_store` | vector files the segments and rebuilds read |
| `ENGINE_EMBED_MODEL` | `qwen3-embedding-8b` | the model pages and queries are embedded with, from [`desearch/embedding.py`](../desearch/embedding.py) |
| `OPENROUTER_API_KEY` | | embeds queries |
| `UNIFIED_ACCESS_KEY_FILE` | `/opt/tc/.index_access_key` | the key callers from other machines must send as `Access-Key` |
| `CF_R2_ENDPOINT`, `CF_R2_ACCESS_KEY_ID`, `CF_R2_SECRET_ACCESS_KEY`, `CF_R2_PAGES_BUCKET` | | read access to published pages and vectors, for `sync` |

## API

`POST /v1/search`

```json
{"query": "...", "mode": "balanced", "count": 10,
 "start_date": "2026-09-01", "end_date": "2026-09-15",
 "highlights": true, "page_text": false}
```

Returns ranked results with `url`, `title`, `published_date`, the closest passage when `highlights`
is set and the page text when `page_text` is set. A date filter leaves out pages with no known date.

`POST /v1/document` with `{"url": "..."}` returns one indexed page. `GET /healthz` reports that the
service is up.

## Tests

```bash
pip install -r engine/requirements.txt pytest
cd engine && pytest
```
