# SN22 Validator API Reference

`neurons/validators/api.py` runs a FastAPI service beside an SN22 validator. Trusted Desearch services use it to request organic subnet-backed search and to inspect public miner state observed by that validator.

The default server URL in OpenAPI is `http://localhost:8005`; override the runtime port with `PORT`.

## Authentication

Protected routes require an `access-key` header that exactly matches the validator's `EXPECTED_ACCESS_KEY` environment variable.

```bash
-H 'access-key: <EXPECTED_ACCESS_KEY>'
```

If `EXPECTED_ACCESS_KEY` is not configured, protected routes return `403`. If the header is wrong or missing, they return `401`.

Public miner status routes under `/public/miners` do not require authentication.

## Request models

### `SearchRequest`

Used by `POST /search`.

| Field | Type | Notes |
| --- | --- | --- |
| `prompt` | string | Search prompt/query. |
| `tools` | string[] | Tool names such as `Twitter Search`, `Web Search`, `ArXiv Search`, `Wikipedia Search`, `Youtube Search`, `Hacker News Search`, `Reddit Search`. |
| `start_date`, `end_date` | string \| null | Optional UTC timestamps, e.g. `2025-05-01T00:00:00Z`. |
| `date_filter` | enum \| null | Defaults to the server's past-week filter. |
| `model` | enum \| null | Defaults to `NOVA`. |
| `count` | integer | Per-source result count. Min `10`, max `200`; default `10`. |
| `result_type` | enum \| null | Defaults to links plus final summary. |
| `system_message` | string \| null | Optional summary instructions. |
| `scoring_system_message` | string \| null | Optional scoring instructions. |
| `chat_history` | object[] | Optional chat history items. |

### `LinksSearchRequest`

Used by the link-only endpoints.

| Field | Type | Notes |
| --- | --- | --- |
| `prompt` | string | Search prompt/query. |
| `tools` | string[] | Used by `/search/links/web` and `/search/links/twitter`; `/search/links` uses the server's full tool list. |
| `model` | enum \| null | Defaults to `NOVA`. |
| `count` | integer | Per-source result count. Min `10`, max `200`; default `10`. |

## Protected routes

| Method | Route | Input | Response |
| --- | --- | --- | --- |
| `GET` | `/` | none | Health check: `{"status":"healthy","version":"..."}` when the validator service is reachable. |
| `POST` | `/search` | `SearchRequest` JSON body | Server-sent event stream of search results. |
| `POST` | `/search/links/web` | `LinksSearchRequest` JSON body | JSON object of aggregated link results for the requested tools. |
| `POST` | `/search/links/twitter` | `LinksSearchRequest` JSON body | JSON object of aggregated X/Twitter link results for the requested tools. |
| `POST` | `/search/links` | `LinksSearchRequest` JSON body | JSON object of aggregated link results across all server-supported tools. |
| `POST` | `/twitter/search` | Twitter filter JSON body | List of matching tweet objects. |
| `POST` | `/twitter/urls` | `{ "urls": ["https://x.com/.../status/..."] }` | List of tweet objects for the requested URLs, or `404` if none are found. |
| `GET` | `/twitter/{id}` | path parameter `id` | One tweet object for the requested tweet ID, or `404` if not found. |

The exact current link endpoint strings in `neurons/validators/api.py` are `/search/links/web`, `/search/links/twitter`, and `/search/links`.

### Link endpoint examples

```bash
curl -s 'http://localhost:8005/search/links/web' \
  -H 'content-type: application/json' \
  -H 'access-key: <EXPECTED_ACCESS_KEY>' \
  -d '{
    "prompt": "latest Bittensor subnet news",
    "tools": ["Web Search", "Hacker News Search"],
    "count": 10
  }'
```

```bash
curl -s 'http://localhost:8005/search/links/twitter' \
  -H 'content-type: application/json' \
  -H 'access-key: <EXPECTED_ACCESS_KEY>' \
  -d '{
    "prompt": "SN22 Desearch",
    "tools": ["Twitter Search"],
    "count": 10
  }'
```

```bash
curl -s 'http://localhost:8005/search/links' \
  -H 'content-type: application/json' \
  -H 'access-key: <EXPECTED_ACCESS_KEY>' \
  -d '{
    "prompt": "decentralized search infrastructure",
    "tools": ["Web Search"],
    "count": 10
  }'
```

`/search/links` currently ignores `tools` for source selection and uses the server's full `available_tools` list.

### Twitter filter search

`POST /twitter/search` accepts:

| Field | Type | Notes |
| --- | --- | --- |
| `query` | string | Search text. Defaults to empty string. |
| `sort` | string | Defaults to `Top`. |
| `user` | string \| null | Optional author filter. |
| `start_date`, `end_date` | string \| null | Optional date filters. |
| `lang` | string \| null | Optional language filter. |
| `verified`, `blue_verified`, `is_quote`, `is_video`, `is_image` | boolean \| null | Optional tweet/account filters. |
| `min_retweets`, `min_replies`, `min_likes` | integer \| null | Optional engagement thresholds. |
| `count` | integer | Max `100`; default `20`. |

## Public miner status routes

| Method | Route | Response |
| --- | --- | --- |
| `GET` | `/public/miners` | Validator identity plus active miners grouped by hotkey and per-search-type state. |
| `GET` | `/public/miners/{hotkey}` | Per-miner state plus 72-hour scoring-window history for each search type. |

Public miner state is read from `MINER_DB_PATH`, which defaults to `.state/miner_state.db` under the repository root.

## OpenAPI routes

FastAPI also serves interactive docs and schema routes by default:

- `GET /docs`
- `GET /openapi.json`

Use these locally to inspect the exact schema generated by the current code.
