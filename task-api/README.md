# Task API

Hands tasks to miners, has validators check the results, and publishes verified work to object
storage.

```
feeder ──URLs──▶ task API ──lease──▶ miner ──upload──▶ temp bucket
                   │  ▲                                    │
          check job│  │verdict                             │
                   ▼  │                                    │
                 validator ◀───────────────────────────────┘
                   │
         pass ──▶ publisher ──▶ pages bucket
```

| Path | |
| --- | --- |
| `app/` | the API: auth, queues, rounds, upload links, verdicts, audits, budgets and shares |
| `publisher/` | writes verified pages and vectors to the pages bucket |
| `feeder/` | sends the bot's newest URLs to the API |
| `tools/verify_round.py` | checks a closed round against its commitment and signed log |

The miner and validator are in [`neurons/`](../neurons/), and the code they share with the API
(text extraction, the signed client, the credit rules) is in [`desearch/`](../desearch/).

## How it works

**Rounds.** URLs are enqueued in rounds. The API packs them into batches, publishes a hash of the
batches, and commits to a block ten blocks ahead; that block's hash sets the order the batches are
served in.

**Leases.** A miner leases a task and receives its URLs and an upload link for one key. Miners never
hold storage credentials. On completion the API copies the upload to a key only it can write, so
nothing the miner changes afterwards is scored. A miner is never given a batch it held before.

**Checks.** A validator leases the completed task and:

- re-extracts a sample of rows from the uploaded HTML;
- fetches a sample of pages itself and compares their text with the miner's, using ScrapingDog only
  for pages its own address cannot load;
- checks a sample of the rows the miner reported as failed.

It returns what it found for each sampled URL. The API works out the credit from those outcomes and
ignores any credit a validator claims.

**Audits.** A share of verdicts is checked again by a second validator, and a third breaks a tie. A
validator that loses too many audits stops receiving jobs.

**Publishing.** A passing task's pages are written to the pages bucket, one object per URL, only when
the content changed and never over a newer fetch.

**Signed log.** Every lease, refusal and completion is signed by the API. `GET /v1/key` publishes the
signer, and `tools/verify_round.py` checks a closed round end to end.

## Scoring

A crawl task fails when it:

- has a row for a URL it was not given, or the same URL twice;
- returns under 85% of its URLs;
- has over 20% of checked rows that do not re-extract exactly from the uploaded HTML;
- has sampled pages whose text does not match the validator's own fetch;
- is over half error rows for pages the validator could load.

A task where nothing could be judged is void: nothing is paid or charged, and it goes back in the
queue. A passing task is credited row by row at the rate its sample checked out.

Each miner has a budget, the number of tasks it may hold from lease to verdict. It grows by one for
each task credited for at least 85% of its URLs and halves on a failure, an expired lease or an
abandoned task. Two failures within 24 hours, if they are at least 5% of the miner's verdicts, lock it
out for 12 hours. Problems on the validator's side never count against a miner.

`GET /v1/shares` returns each miner's share of the credited work over the last 24 hours, per pool.
How much of the emission each pool gets is set in the validator (`POOLS` in
[`neurons/validators/weights.py`](../neurons/validators/weights.py)). See
[Emission](../docs/emission.md).

## Embed tasks

Embed tasks are off (`TASK_API_EMBED_TASKS=0`) until Desearch's embedding model ships. When on, the
publisher cuts each batch of new or changed pages into texts (a head, the full text and each
passage), and the API opens one embed task per 500 pages. The miner returns one vector per text. The
validator checks every vector's shape and recomputes 20 of them with the same model; each must reach
a cosine similarity of 0.99. A pass is credited with the characters embedded.

The API records which version of each page every model has embedded, so an unchanged page is not
embedded twice. The miner-facing description is [Embedding tasks](../docs/embedding-tasks.md).

## Run

The API host runs the API, its Redis and the publisher:

```bash
cp deploy/.env.example .env   # R2 buckets and credentials, admin hotkeys, signing key
docker compose --env-file .env -f deploy/docker-compose.prod.yml up -d --build
curl -s localhost:8080/v1/health
```

It needs two R2 buckets in the same jurisdiction, with a token that can read and write both:

- a temp bucket for uploads, with a lifecycle rule that deletes objects after one day;
- a pages bucket for published work, kept permanently.

The feeder runs next to the bot, where its stores are:

```bash
PYTHONPATH=.. python -m feeder --buckets /mnt/desearch-bot/buckets --domains feeder/news_domains.json
```

| Variable | Default | |
| --- | --- | --- |
| `TASK_API_REGISTRY` | — | `chain` or `local` |
| `TASK_API_ADMIN_HOTKEYS` | — | hotkeys allowed to enqueue |
| `TASK_API_KEY_URI` | — | the key the log is signed with; required in `chain` mode |
| `TASK_API_LEASE_TTL`, `TASK_API_VALIDATION_TTL` | 900 | seconds a lease lasts |
| `TASK_API_AUDIT_RATE` | 0.05 | share of verdicts audited |
| `TASK_API_MAX_ATTEMPTS` | 3 | times a task is retried before it is dropped |
| `TASK_API_EMBED_TASKS` | 0 | 1 opens embed tasks |
| `TASK_API_EMBED_MODEL` | `qwen3-embedding-8b` | the model embed tasks name, from [`desearch/embedding.py`](../desearch/embedding.py) |

In `chain` mode a validator needs a validator permit and 1000 stake.

## Public endpoints

| Endpoint | |
| --- | --- |
| `GET /v1/health` | queue depths, backlog and every validator's audit record |
| `GET /v1/shares` | every miner's share, per pool |
| `GET /v1/tasks` | scored tasks, newest first; filter with `miner`, `validator`, `since` |
| `GET /v1/tasks/{task_id}` | one task's state and verdict, with per-URL detail for a week |
| `GET /v1/miners/{hotkey}` | a miner's budget, coverage, verdicts and lockout |
| `GET /v1/rounds`, `GET /v1/rounds/{id}` | round commitments |
| `GET /v1/rounds/{id}/log` | a round's signed log |
| `GET /v1/key` | the log's signer |

## Storage

```
temp bucket, emptied after a day
  uploads/        miner uploads
  submitted/      the frozen copies validators score
  embed-inputs/   texts waiting to be embedded
pages bucket, permanent
  pages/<domain>/<sha1>   the latest verified version of each URL, zstd JSON
  changes/                every new or changed page, for the index to follow
  reports/                every final verdict, with its samples and votes
  vectors/model=<name>/   verified vectors
```

A page's key comes from its URL alone:

```python
from app.canonical import canonicalize
from publisher.records import page_key

key = page_key(canonicalize(url))   # pages/<domain>/<sha1>
```

## Tests

```bash
cd task-api && python3 -m pytest -q
```

The tests use Redis db 14 and an in-memory R2; run one suite at a time. `TASK_API_TEST_R2=1` also runs
the storage tests against real buckets, under `_test/`.
