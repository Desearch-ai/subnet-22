# Task API

Hands tasks to miners, lists a drawn share of their uploads for validators to check, and publishes
the work to object storage.

```
bot ──URLs──▶ task API ──claim──▶ miner ──upload──▶ subnet-22 bucket (public)
 ▲               ▲  │                                        │
 │  pass or fail │  └── writes the open list and the notes ─▶│
 │               │                                           ▼ reads the drawn uploads
 │               └──────────────────────────────────────── validator ──re-fetch──▶ the web
 │               │
 │     pass ──▶ publisher ──▶ desearch-pages bucket
 │                  │
 └── outcome feed ◀─┘
```

| Path | |
| --- | --- |
| `app/` | the API: auth, queues, rounds, upload links, check results, budgets and shares |
| `feeder/` | `feeder.sample` exports the URL dataset the miner sandbox serves |
| `tools/verify_round.py` | checks a closed round against its commitment and signed log |

The miner and validator are in [`neurons/`](../neurons/), and the code they share with the API
(text extraction, the signed client, the payment rules) is in [`desearch/`](../desearch/). The publisher,
which writes passed pages and vectors to the pages bucket and takes withdrawn ones down, is in
[`publisher/`](../publisher/).

## How it works

**Queue.** The [bot](../desearch-bot/README.md) fills the queue: it asks `GET /v1/room` how many
tasks the queue can take and enqueues that many, each batch with an id so a batch sent twice is
queued once. It reads back what became of every URL from the outcome feed.

**Rounds.** URLs are enqueued in rounds. The API packs them into tasks of 1,000 URLs, the last one
of a round taking whatever is left, publishes a hash of the batches, and commits to a block ten
blocks ahead; that block's hash sets the order the batches are served in.

**Claims.** A miner asks for as many tasks as it can take and receives, for each, its URLs and an
upload link for one key; it never holds storage credentials. The claim's time starts when the API
hands the tasks over. On completion the API copies the upload to a key only it can write, so nothing
changed afterwards is scored. A completion counts as on time when it reaches the API inside the
claim, however long the API then takes over it; a file written after the completion reached the API
is refused. Sending the same completion again returns the first one's answer. A miner is never given
a batch it held before.

**Which uploads are checked.** On completion the API freezes the upload and writes a signed note
beside it (the task's URLs and the freeze block). Once the block ten blocks after the freeze exists,
its hash decides whether validators check the upload, so no miner can know while uploading. Checked
are:

- every upload of a hotkey until it has passed 10 checks;
- the next 10 uploads of a hotkey after a failed check;
- every upload of a hotkey that is locked out, or whose completion report gives no row counts;
- otherwise a share of each hotkey's uploads, and about one an hour at least.

**Upload log.** Every completed crawl upload, with the counts its miner reported, is written every
30 seconds to a numbered file signed by the API, `log/uploads/seq/<n>.json` in the uploads bucket,
with the newest number in `log/uploads/latest.json`. Validators count each miner's work from it and
set their weights themselves.

**Checks.** A drawn upload is listed in `validation/open.json` in the same public bucket, so
validators read everything without asking the API. The rows to check are picked with the same block
hash, so every validator checks the same rows. Each validator:

- re-extracts the picked rows from the uploaded HTML;
- fetches the picked pages itself and compares their text with the miner's, using ScrapingDog only
  for pages its own address cannot load;
- checks the picked rows among those the miner reported as failed.

It reports what it found for each checked URL, once per upload; the API works out the paid rows from
those findings and ignores any number a validator states. Reports stay sealed until finalized.

**Final verdict.** A checked upload is finalized when every active validator has reported, or at
its deadline on the reports it has; a validator is active for an hour after its last report. One
active validator finalizes alone. An upload no validator reported on by its deadline is finalized
like an unchecked one.

The majority decides pass or fail, and a pass pays the miner's rows at the rate the checked pages
matched (the lower middle count when the majority differs). A validator that disagrees with the
majority is marked, as are two passes more than 15% apart on the rows to pay; one that keeps
disagreeing stops receiving uploads.

**Unchecked uploads.** An upload no check drew is finalized as soon as its block hash exists, on the
counts the miner reported at completion: its content rows, and its error rows at the share of the
hotkey's reported errors that checks of the last 3 days could reproduce. A report under 85% of the
task's URLs fails.

**Take-back.** A failed check takes back everything the hotkey passed since its last passed check:
the pay and the published pages. Two failed checks among a hotkey's last 10 also cost it its pay of
the last 24 hours, its pages of the last 24 hours and a 48-hour lockout. A checked upload whose
report claims more content rows than its file holds fails.

**Publishing.** The publisher reads a passed upload's text columns in byte ranges, never its HTML,
and refuses a file whose footer promises more than its task could hold. It keeps its own index of
every page's latest version, so a page is published only when its content changed and never over a
newer fetch. Each batch's new, changed and removed pages, with their full records, go into one change
file in the pages bucket, numbered like the outcome feed: `changes/seq/<n>.json` names the file and
`changes/latest.json` holds the newest number. The index notes which change file and row hold every
page's latest version. A withdrawn upload's pages are recorded as removed while they are still its
version.

**Outcome feed.** Every URL ends as published, unchanged, failed or dropped. The API and the
publisher write these in numbered files, `outcomes/seq/<n>.json` in the uploads bucket each naming a
Parquet file, with the newest number in `outcomes/latest.json`. A task dropped after its last
attempt and a withdrawn page are reported dropped, so the bot sends them again.

**Signed log.** Every claim, refusal and completion is signed by the API. `GET /v1/key` publishes the
signer, and `tools/verify_round.py` checks a closed round end to end.

## Scoring

A crawl task fails when it:

- has a row for a URL it was not given, or the same URL twice;
- returns under 85% of its URLs;
- has over 20% of checked rows that do not re-extract exactly from the uploaded HTML;
- has sampled pages whose text does not match the validator's own fetch;
- is over half error rows for pages the validator could load.

A checked task where nothing could be judged decides nothing: nothing is paid or charged, and it goes back
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

**Budgets and lockouts.**

- A miner may crawl as many tasks at once as its budget, with up to twice that waiting for a verdict.
- The budget starts at 1, grows by half for each task paid for at least 85% of its URLs, up to 100,
  and halves on a failure, an expired claim or an abandoned task.
- Each of those is also a strike and takes the task's URLs back from the miner's paid rows; lapses
  within 5 minutes of each other are one strike.
- Two strikes within 24 hours, if at least 5% of its checked tasks, lock the miner out for an hour,
  then 12 hours, then 48 within a week. An upload that crashes most validators' checks locks it out
  for a week. Validator-side problems never count.

A completed upload must be Parquet: the API checks its first and last bytes and never parses it.
Failed and void uploads are deleted. A miner may claim every 2 seconds; a refusal repeated within a
minute is not signed into the log again.

`GET /v1/shares` returns each miner's share of the paid work over the last 24 hours, per pool.
How much of the emission is burned and how the rest is split between the pools is set in the
validator (`EMISSION_CONTROL_PERC`, `CRAWL_PERC` and `EMBED_PERC` in
[`neurons/validators/weights.py`](../neurons/validators/weights.py)). See
[Emission](../docs/emission.md).

## Embed tasks

Embed tasks are off (`TASK_API_EMBED_TASKS=0`) until Desearch's embedding model ships and the
publisher writes their inputs; it refuses to start while they are on. Each embed task holds the
texts of 500 new or changed pages (a head, the full text and each passage). The miner returns one
vector per text. The validator checks every vector's shape and recomputes 20 of them with the same
model; each must reach a cosine similarity of 0.99. A pass pays the characters embedded.

The API records which version of each page every model has embedded, so an unchanged page is not
embedded twice. The miner-facing description is [Embedding tasks](../docs/embedding-tasks.md).

## Run

The API is built from `Dockerfile.rust` at the repository root. It keeps its SQLite file in `/data`
and needs a Redis (`TASK_API_REDIS`); the [publisher](../publisher/) runs on its own machine:

```bash
docker build -f task-api/Dockerfile.rust -t task-api .
docker run -d --env-file task-api/deploy/.env.example -e TASK_API_REGISTRY=chain -e TASK_API_SEEDS=chain -v task-api-data:/data -p 127.0.0.1:8080:8080 task-api
curl -s localhost:8080/v1/ping
```

Fill in the R2 credentials, admin hotkeys, signing key and Redis URL in the env file first. Behind a
proxy that checks `/v1/ping`, a deploy can roll: the new copy serves next to the old one until the old
one stops, and only one copy at a time runs the background work.

It needs two R2 buckets in the same jurisdiction, with a token that can read and write both:

- `subnet-22`, the temporary bucket for uploads, with a lifecycle rule that deletes objects after one day;
- `desearch-pages`, the permanent bucket for published work.

Validators read `subnet-22` directly, so serve it publicly on a custom domain: R2 → the bucket →
Settings → Public access → Custom Domains (or `wrangler r2 bucket domain add subnet-22 --domain
<hostname>`), not the rate-limited `r2.dev` URL. Objects are readable but not listable, which is why
the API keeps `validation/open.json`. Mainnet serves it at `https://r2.desearch.ai`, the validator's
default `--neuron.storage_url`.

The [bot](../desearch-bot/README.md) fills the queue with the admin hotkey (`--task-api`).

| Variable | Default | |
| --- | --- | --- |
| `TASK_API_REGISTRY` | — | `chain` or `local` |
| `TASK_API_SEEDS` | `local` | `chain` or `local`: where round seeds come from |
| `TASK_API_REDIS` | `redis://localhost:6379/15` | the Redis URL |
| `TASK_API_LISTEN` | `0.0.0.0:8080` | the address the API listens on |
| `TASK_API_ADMIN_HOTKEYS` | — | hotkeys allowed to enqueue |
| `TASK_API_KEY_URI` | — | the key the log is signed with; required in `chain` mode |
| `TASK_API_CLAIM_TTL` | 180 | seconds a miner is given to upload a task after claiming it; the claim and its upload link last 10 seconds longer |
| `TASK_API_POLL_RATE` | 0.5 | claim requests per second one hotkey may make |
| `TASK_API_VALIDATION_TTL` | 900 | seconds validators have to report once the upload is open |
| `TASK_API_ACTIVE_S` | 3600 | seconds since its last report a validator counts as active |
| `TASK_API_LEDGER_DELAY_S` | 0 | seconds finalized uploads and shares stay out of public view |
| `TASK_API_MAX_ATTEMPTS` | 3 | times a task is retried before it is dropped |
| `TASK_API_CHECK_SHARE` | `SHARE` in [`app/sampling.py`](app/sampling.py) | share of an established hotkey's uploads drawn for a check; 1 checks every upload |
| `TASK_API_QUEUE_TARGET` | 1200 | tasks the bot keeps queued and waiting for their round's reveal |
| `TASK_API_READS_PER_MINUTE` | 120 | `GET` requests one IP may make in a minute, outside the log endpoints |
| `TASK_API_LOG_READS_PER_MINUTE` | 60 | log requests (tasks, votes, miners, validators, overview) one IP may make in a minute |
| `TASK_API_FAILED_WRITES_PER_MINUTE` | 30 | failed sign-ins after which one IP's writes are refused for the rest of the minute |
| `TASK_API_CORS_ORIGINS` | local dev servers | comma-separated origins whose pages may read the API from a browser |
| `TASK_API_EMBED_TASKS` | 0 | 1 opens embed tasks |
| `TASK_API_EMBED_MODEL` | `qwen3-embedding-8b` | the model embed tasks name, from [`desearch/embedding.py`](../desearch/embedding.py) |

In `chain` mode a validator needs a validator permit, at least 10,000 total stake and at least 20
alpha of its own.

## Public endpoints

Anyone can read these; the [UI](../ui/README.md) is built on them.

| Endpoint | |
| --- | --- |
| `GET /v1/overview` | the queue, the work in progress and the totals of the window |
| `GET /v1/live` | the tasks miners hold now and the uploads being checked, with who has voted |
| `GET /v1/stats/series` | tasks, verdicts and rows per 5, 15 or 60 minutes; for everyone, one `miner` or one `validator` |
| `GET /v1/tasks` | checked tasks with the rows each paid, newest first; filter with `miner`, `validator`, `verdict`, `kind`, `since` |
| `GET /v1/tasks/{task_id}` | one task's state and result, every validator's vote, and per-URL detail for 2 days |
| `GET /v1/votes` | every validator's vote on every checked upload; filter with `validator`, `miner`, `task_id`, `verdict`, `agreed` |
| `GET /v1/miners` | every miner's budget, coverage, share and results in the window |
| `GET /v1/miners/{hotkey}` | a miner's budget and its history, coverage, share, pass and fail counts and lockout |
| `GET /v1/miners/{hotkey}/verdicts` | signed by that miner: its own finalized uploads, without the ledger delay |
| `GET /v1/validators` | every validator's votes, how often it agreed with the final result, and its standing |
| `GET /v1/validators/{hotkey}` | the same for one validator |
| `GET /v1/events` | the signed log as a feed: tasks issued, completed, refused and returned; filter with `miner`, `task_id`, `outcome` |
| `GET /v1/health` | queue depths, backlog, the active validators and every validator's standing |
| `GET /v1/room` | how many tasks the queue can take now, for the bot |
| `GET /v1/shares` | every miner's share of the paid work as the API counts it, per pool |
| `GET /v1/rounds`, `GET /v1/rounds/{id}` | round commitments |
| `GET /v1/rounds/{id}/log` | a round's signed log |
| `GET /v1/key` | the key that signs the log and the notes next to uploads |

Every miner and validator is named by hotkey, with its uid beside it (`miner_uid`, `validator_uid`,
`uid`); the uid is null for a hotkey that is not on the metagraph.

Lists return a page at a time (`limit`, at most 100) with a `next` value to pass back as `before`.

**Limits.**

- Reads are limited per IP, with a separate budget for the log endpoints; over it, `429` with
  `Retry-After`.
- An IP that keeps failing to sign in has its writes refused for the rest of the minute.
- A body over 64 kB (16 MB for a verdict or an enqueue) is refused with `413` before it is read.
- Log reads have their own database connection and thread, so they never delay claims or verdicts;
  when too many wait, the API answers `503` with `Retry-After`.
- The caller's address is taken from Cloudflare's `CF-Connecting-IP` header, so serve the API only
  through Cloudflare.

## Storage

```
subnet-22 bucket, temporary, emptied after a day, public
  uploads/              miner uploads
  submitted/            the frozen copies validators check, each with a signed .manifest.json beside it
  validation/open.json  the uploads waiting for validators, with their manifests, rewritten as they change
  log/uploads/          every completed upload with its reported counts, numbered and signed
  outcomes/             what became of every URL, numbered under seq/, newest in latest.json
  embed-inputs/         texts waiting to be embedded
desearch-pages bucket, permanent
  changes/                every new, changed or removed page with its full record, naming the miner
                          that crawled it and the validator that checked it; numbered under seq/,
                          newest in latest.json
  index/snapshots/        a daily copy of the publisher's version index
  reports/                every final result, with the checked pages and each validator's report
  vectors/model=<name>/   verified vectors
```

A page's key comes from its URL alone; it is how the index and the change files name a page:

```python
from app.canonical import canonicalize, domain_of, url_sha1

url = canonicalize(assigned_url)
key = f"pages/{domain_of(url)}/{url_sha1(url)}"
```

## Tests

```bash
cd task-api && python3 -m pytest -q
```

The tests use Redis db 14 and an in-memory R2; run one suite at a time. `TASK_API_TEST_R2=1` also runs
the storage tests against real buckets, under `_test/`.
