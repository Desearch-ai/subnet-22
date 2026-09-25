# Task API

Hands tasks to miners, has validators check the results, and publishes verified work to object
storage.

```
feeder ──URLs──▶ task API ──claim──▶ miner ──upload──▶ subnet-22 bucket
                   │  ▲                                    │
          check job│  │result                              │
                   ▼  │                                    │
                 validator ◀───────────────────────────────┘
                   │
         pass ──▶ publisher ──▶ desearch-pages bucket
```

| Path | |
| --- | --- |
| `app/` | the API: auth, queues, rounds, upload links, check results, budgets and shares |
| `publisher/` | writes verified pages and vectors to the pages bucket |
| `feeder/` | sends the bot's newest URLs to the API |
| `tools/verify_round.py` | checks a closed round against its commitment and signed log |

The miner and validator are in [`neurons/`](../neurons/), and the code they share with the API
(text extraction, the signed client, the payment rules) is in [`desearch/`](../desearch/).

## How it works

**Rounds.** URLs are enqueued in rounds. The API packs them into batches, publishes a hash of the
batches, and commits to a block ten blocks ahead; that block's hash sets the order the batches are
served in.

**Claims.** A miner claims a task and receives its URLs and an upload link for one key. Miners never
hold storage credentials. On completion the API copies the upload to a key only it can write, so
nothing the miner changes afterwards is scored. A miner is never given a batch it held before.

**Checks.** On completion the API records the chain block it froze the upload at, writes a signed
note next to the frozen copy with the task's URLs and that block, and lists the upload in
`validation/open.json` in the same bucket. The bucket is public, so validators read the list and
the uploads without asking the API. The rows to check are picked with the hash of the block ten
blocks after the freeze, which validators take from the chain, so no miner can know them while
uploading and every validator checks the same rows. Each validator:

- re-extracts the picked rows from the uploaded HTML;
- fetches the picked pages itself and compares their text with the miner's, using ScrapingDog only
  for pages its own address cannot load;
- checks the picked rows among those the miner reported as failed.

It reports what it found for each checked URL, once per upload. The API works out from those
findings how many rows the miner is paid for, and ignores any number a validator states. Reports
stay sealed until the upload is finalized.

**Final verdict.** An upload is finalized when every active validator has reported, or at its
deadline with more than half of them; active means having asked for work or reported within the
last hour. A validator that is still checking an upload therefore holds it until it reports or the
deadline passes. The majority decides pass or fail. A passing upload pays the miner for its rows at
the rate the checked pages matched; when the validators in the majority arrived at different counts,
the lower middle value is paid. Validators that disagreed with the majority are marked, and two
validators that both pass an upload but differ by more than 15% on the rows to pay count as
disagreeing. One active validator finalizes alone. Below the quorum at the deadline nothing is
decided and the task goes back out. A validator that keeps disagreeing with the majority stops
receiving uploads.

**Publishing.** A passing task's pages are written to the pages bucket, one object per URL, only when
the content changed and never over a newer fetch.

**Signed log.** Every claim, refusal and completion is signed by the API. `GET /v1/key` publishes the
signer, and `tools/verify_round.py` checks a closed round end to end.

## Scoring

A crawl task fails when it:

- has a row for a URL it was not given, or the same URL twice;
- returns under 85% of its URLs;
- has over 20% of checked rows that do not re-extract exactly from the uploaded HTML;
- has sampled pages whose text does not match the validator's own fetch;
- is over half error rows for pages the validator could load.

A task where nothing could be judged decides nothing: nothing is paid or charged, and it goes back
in the queue. The same goes for a task whose checked rows the validator could mostly not read, and
for a check the validator could not finish because of its own timeout or crash. A passing task pays
the miner row by row at the rate its checked pages matched.

Rows a check turned down are not published even when the task passes: a sampled page whose text
differs from the validator's fetch, a row whose text does not re-extract from its HTML, and a page
whose title or author differs from the live page. A sampled page whose publication date or canonical
URL differs from the live page counts as a mismatch, since those never change between two fetches.

The API takes a validator's findings per page, not its counts, on trust: a report whose rows the
task could not have produced, such as more checked rows than rows returned or a checked URL outside
the task, is refused.

Each miner has a budget, the number of tasks it may hold from claim to final verdict. It grows by
one for each task paid for at least 85% of its URLs and halves on a failure, an expired claim or an
abandoned task. Two failures within 24 hours, if they are at least 5% of the miner's checked tasks,
lock it out for 12 hours. Problems on the validator's side never count against a miner.

`GET /v1/shares` returns each miner's share of the paid work over the last 24 hours, per pool.
How much of the emission each pool gets is set in the validator (`POOLS` in
[`neurons/validators/weights.py`](../neurons/validators/weights.py)). See
[Emission](../docs/emission.md).

## Embed tasks

Embed tasks are off (`TASK_API_EMBED_TASKS=0`) until Desearch's embedding model ships. When on, the
publisher cuts each batch of new or changed pages into texts (a head, the full text and each
passage), and the API opens one embed task per 500 pages. The miner returns one vector per text. The
validator checks every vector's shape and recomputes 20 of them with the same model; each must reach
a cosine similarity of 0.99. A pass pays the characters embedded.

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

- `subnet-22`, the temporary bucket for uploads, with a lifecycle rule that deletes objects after one day;
- `desearch-pages`, the permanent bucket for published work.

Validators read `subnet-22` directly, so it is served publicly: in the Cloudflare dashboard, R2 →
the bucket → Settings → Public access → Custom Domains, connect a hostname on a zone of the same
account (or `wrangler r2 bucket domain add subnet-22 --domain <hostname>`). Objects are then
readable at `https://<hostname>/<key>`; listing is not, which is why the API keeps
`validation/open.json`. Validators are started with that URL as `--neuron.storage_url`. The
`r2.dev` development URL is rate limited and not meant for this.

The feeder runs next to the bot, where its stores are:

```bash
PYTHONPATH=.. python -m feeder --buckets /mnt/desearch-bot/buckets --domains feeder/news_domains.json
```

| Variable | Default | |
| --- | --- | --- |
| `TASK_API_REGISTRY` | — | `chain` or `local` |
| `TASK_API_ADMIN_HOTKEYS` | — | hotkeys allowed to enqueue |
| `TASK_API_KEY_URI` | — | the key the log is signed with; required in `chain` mode |
| `TASK_API_CLAIM_TTL` | 900 | seconds a miner's claim lasts |
| `TASK_API_VALIDATION_TTL` | 900 | seconds validators have to report once the upload is open |
| `TASK_API_ACTIVE_S` | 3600 | seconds since its last report a validator counts as active |
| `TASK_API_LEDGER_DELAY_S` | 0 | seconds finalized uploads and shares stay out of public view |
| `TASK_API_MAX_ATTEMPTS` | 3 | times a task is retried before it is dropped |
| `TASK_API_EMBED_TASKS` | 0 | 1 opens embed tasks |
| `TASK_API_EMBED_MODEL` | `qwen3-embedding-8b` | the model embed tasks name, from [`desearch/embedding.py`](../desearch/embedding.py) |

In `chain` mode a validator needs a validator permit and 1000 stake.

## Public endpoints

| Endpoint | |
| --- | --- |
| `GET /v1/health` | queue depths, backlog, the active validators and every validator's standing |
| `GET /v1/shares` | every miner's share, per pool |
| `GET /v1/tasks` | checked tasks with the rows each paid, newest first; filter with `miner`, `validator`, `since` |
| `GET /v1/tasks/{task_id}` | one task's state and result, with per-URL detail for a week |
| `GET /v1/miners/{hotkey}` | a miner's budget, coverage, pass and fail counts and lockout |
| `GET /v1/miners/{hotkey}/verdicts` | signed by that miner: its own finalized uploads, without the ledger delay |
| `GET /v1/rounds`, `GET /v1/rounds/{id}` | round commitments |
| `GET /v1/rounds/{id}/log` | a round's signed log |
| `GET /v1/key` | the log's signer |

## Storage

```
subnet-22 bucket, temporary, emptied after a day, public
  uploads/              miner uploads
  submitted/            the frozen copies validators check, each with a signed .manifest.json beside it
  validation/open.json  the uploads waiting for validators, with their manifests, rewritten as they change
  embed-inputs/         texts waiting to be embedded
desearch-pages bucket, permanent
  pages/<domain>/<sha1>   the latest verified version of each URL, zstd JSON, naming the miner
                          that crawled it and the validator that checked it
  changes/                every new or changed page, for the index to follow
  reports/                every final result, with the checked pages and each validator's report
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
