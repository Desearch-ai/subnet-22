# Desearch Task API

Hands crawl tasks to miners, stores their results in R2, and has validators score them.

```
seed ──> api ──lease──> miner ──PUT parquet──> subnet-22/uploads/
          │                                          │
          ├──complete: copy to subnet-22/submitted/ <┘
          │                     │
          ├──score── validator <┘ presigned GET, re-fetches a sample itself, ScrapingDog as fallback
          │
          └──pass──> publisher ──> desearch-pages/pages/<domain>/<sha1>, one object per URL
```

| Path | What |
| --- | --- |
| `app/` | the API: auth, Redis queue, rounds, R2 presigning, validation and audits, budgets, URL canonicalization |
| `publisher/` | writes verified text per URL into the pages bucket, only when it changed |
| `feeder/` | reads desearch-bot's stores and keeps the queue fed (`python -m feeder`) |
| `tools/` | `verify_round.py`, the public round verifier |

The miner and the validator that talk to this API are in [`neurons/`](../neurons/), and the code
all three share (text extraction, the signed client, the credit rules) is in the
[`desearch`](../desearch/) package at the repository root.

## Run it

The API host runs the API, its Redis and the publisher; miners, validators and the bot feeder run
elsewhere. It needs a `.env` with `CF_R2_BUCKET` (temp), `CF_R2_PAGES_BUCKET` (defaults to
`desearch-pages`), `CF_R2_ENDPOINT`, `CF_R2_ACCESS_KEY_ID`, `CF_R2_SECRET_ACCESS_KEY`,
`TASK_API_ADMIN_HOTKEYS` and `TASK_API_KEY_URI` (see `deploy/.env.example`):

```bash
docker compose --env-file .env -f deploy/docker-compose.prod.yml up -d --build
curl -s localhost:8080/v1/health
```

Buckets, once per account:

1. Create the pages bucket in the same jurisdiction as the temp bucket, and point `CF_R2_ENDPOINT` at
   that jurisdiction's endpoint (`https://<account>.us.r2.cloudflarestorage.com` for US).
2. Give the token Object Read & Write on both buckets.
3. On the temp bucket, and only there, add a lifecycle rule that deletes objects one day after
   upload.

The API refuses to start if either bucket is unreachable. The image is built from the repository
root (see `Dockerfile`), because the API imports the shared `desearch` package.

Miners never see R2 credentials: each lease carries a presigned PUT for exactly one key. On
`/complete` the API copies that object to a key only it can write, so nothing the miner uploads
afterwards reaches the validator or `pages/`.

The bot feeder (`deploy/task-api-feeder.service`) runs beside desearch-bot and reads its bucket
stores, so the queue is fed from what the crawl loop already found. It opens each store as a RocksDB
secondary, which reads a live store without taking its lock, and takes each domain's most recently
listed pages:

```bash
PYTHONPATH=.. python -m feeder --buckets /mnt/desearch-bot/buckets --domains feeder/news_domains.json
```

It needs `rocksdict`, the repository root on `PYTHONPATH` for the shared package, and to run where
the stores are. Which URLs have gone out is kept in its own
`--state` database, not in the crawler's records. A URL goes out again when its lastmod moves, or once
`--refresh` (a day) has passed, so an unchanged article is not paid for every hour, and it counts as
sent only once the API accepted its batch. The feeder skips a cycle while the queue is above
`--queue-cap`, and once the miners drain it below `--low-water` it starts the next cycle instead of
waiting out the interval.

## Miners and validators

Run them from the repository root: see [Miner Setup](../docs/miner-setup.md) and
[Validator Setup](../docs/validator-setup.md).

## API

| Variable | Default | |
| --- | --- | --- |
| `TASK_API_REGISTRY` | — | `chain` or `local`; the API will not start without it |
| `TASK_API_VALIDATOR_URIS` | — | validator hotkey seeds, `local` registry only |
| `TASK_API_ADMIN_URIS`, `TASK_API_ADMIN_HOTKEYS` | — | hotkeys allowed to enqueue |
| `TASK_API_KEY_URI` | `//TaskApi` locally | the secret receipts are signed with; required in `chain` mode |
| `TASK_API_MAX_UPLOAD_BYTES` | 64000000 | larger uploads are refused and removed at `/complete` |
| `TASK_API_VALIDATION_TRIES` | 3 | validation leases that may lapse before the task is voided |
| `TASK_API_VALIDATOR_LEASES` | 8 | validation jobs one validator may hold at once |
| `TASK_API_MAX_RELEASES` | 3 | hand-backs of one job before it is voided as `unjudged` |
| `TASK_API_RELEASES_PER_HOUR` | 60 | hand-backs a validator may make in an hour before its leases are refused |
| `TASK_API_AUDIT_RATE` | 0.05 | share of verdicts checked by a second validator |
| `TASK_API_AUDIT_WAIT_S` | 3600 | how long an audit waits for another validator before the votes so far decide |
| `TASK_API_MAX_ATTEMPTS` | 3 | times a task may be failed or voided before it is dropped |
| `TASK_API_MAX_BACKLOG_S` | 43200 | crawl leases are refused (`VALIDATION_BACKLOG`) while the oldest waiting validation is older |
| `TASK_API_LEASE_TTL`, `TASK_API_VALIDATION_TTL` | 900 | seconds |
| `TASK_API_BLOCK_SECONDS` | 12 | block time of the `local` seeds; tests shorten it |
| `TASK_API_READS_PER_MINUTE` | 120 | public GET requests one client IP may make per minute; more get 429 |
| `TASK_API_POLL_RATE` | 2 | lease requests one hotkey may make per second; more are refused `RATE_LIMITED` |
| `TASK_API_EMBED_MODEL` | `qwen3-embedding-8b` | the model new embed rounds ask for; see `desearch/embedding.py` |

In `chain` mode a validator needs a validator permit and 1000 stake. Validators cannot lease crawl
tasks. The metagraph is refreshed in the background every ten minutes; if a refresh fails, the last
good copy keeps serving.

## Logs

Every verdict carries what the validator found for each URL: its status or error, and for sampled
pages 500-character snippets of both texts and where they diverge. The API keeps that detail for 7
days in its database (`task_api.db` under `TASK_API_DATA`, which also holds budgets, rounds, the
round log and verdicts); the verdict itself, with its samples and votes, stays permanently in the R2
report. Every GET counts against a per-IP budget; the prod compose runs uvicorn with
`--proxy-headers`, so the client IP is the one the reverse proxy in front reports, and the port is
bound to localhost so nothing else can set it:

- `GET /v1/tasks?miner=&validator=&since=&before=&limit=` lists scored tasks, newest first, 50 per page
  by default and at most 100; pass the returned `next` as `before` for the next page
- `GET /v1/tasks/{task_id}` shows one task's state, its last verdict and, for a week, the per-URL detail
- `GET /v1/miners/{hotkey}` shows a miner's budget, coverage, verdicts and budget history

## Rounds

`POST /v1/admin/enqueue` opens a round: it packs the URLs into batches, commits to their hash and to
a seed block ten blocks ahead, and returns. The janitor queues the batches once that block exists,
in the order its hash decides. `GET /v1/rounds` lists recent commitments, so anyone can record a
manifest hash before its seed block exists. Rounds are stored on disk and survive a restart.

Every lease, refusal, completion and hand-back is logged and signed with `TASK_API_KEY_URI`, except
lease requests over `TASK_API_POLL_RATE`, which are refused without a receipt so a flooding hotkey
cannot grow the log. Every refusal carries `retry_after`, the seconds to wait before asking again.
`GET /v1/key` publishes the signer.

Batches are served in the round's order, except that a miner is never given a batch it held before:
it gets the first batch it has not held, and `ALREADY_HELD` if it held every batch near the head.
One miner therefore cannot fail the same batch until it is dropped. A round closes once every task in it is decided, and its log is
then anchored. `tools/verify_round.py --round <id> --signer <ss58>` checks the commitment, the serve
order, the anchored log and every signature, and any receipts or batches you kept.

## Scoring

A task fails on any of:

- a row for a URL it was not assigned, or the same URL twice
- under 85% of assigned URLs returned
- over 20% of checked rows whose text, title, page type, dates, headings or hashes do not re-extract
  exactly from the uploaded HTML
- sampled pages whose text does not match a fresh fetch (thresholds calibrated on real page pairs),
  including text that keeps a page's words but not its figures
- error rows the validator could load making up over half the batch

The validator fetches each sample itself first, from its own IP, exactly as a miner would. ScrapingDog
is used only when that fetch is refused, blocked or not HTML, which keeps the paid fetches for the
pages our own address cannot see. A sample that mismatches or that ScrapingDog fails to load is
fetched once more with rendering before it counts against the miner. The rendered copy can only clear
a sample, never fail one: a miner fetching plain HTTP cannot see what JavaScript adds. ScrapingDog's
own failures (timeouts, 429, 5xx) are not evidence either way. A page neither our IP nor ScrapingDog
could load confirms a miner's error on it. When half the sample could be fetched by neither, the
validator hands the task back, and after three such tasks in a row it pauses: its own network is
the likely problem.

A task where nothing could be judged, no page compared and no error confirmed, is void: nothing is
paid or charged. A failed or void task goes back into the queue at its place, up to
`TASK_API_MAX_ATTEMPTS`. A lapsed or repeatedly handed-back validation voids the task rather than
failing it: a validator's trouble is not the miner's.

A passing task is credited row by row at the rate its sample confirmed: fetched pages times
matched/compared, plus error rows times confirmed/judged. The API works this out from the sample
outcomes a validator submits and ignores any credit it claims. A page a miner could not get but
ScrapingDog could is not paid for, and is not treated as fraud. A pass is published only when at
least one sample was compared.

A miner's budget caps its tasks from lease to verdict, so work waiting for a validator counts too.
The budget grows only when a task is credited for at least 85% of its URLs, and halves on a failure,
an expired lease or an abandoned task.

Two failures the miner caused within 24 hours, if they are at least 5% of its verdicts in that time,
lock it out for 12 hours: lease requests are refused
`LOCKED_OUT` with the time it ends, and `GET /v1/miners/{hotkey}` shows `locked_until`. A failure
counts when its reason is one of the rules above; `unscorable`, the validator's own scorer timing out
or crashing, never does, and neither do expired leases.

Shares are kept per pool, one for each task family. `/v1/shares` returns
`{"window_hours": 24, "pools": {"crawl": {hotkey: share}}}`. A crawl share is a miner's fraction of
the pages credited in the last 24 hours, among miners that returned at least 85% of the URLs
assigned to them over the same 24 hours. How much of the emission each pool gets is set in the
validator (`POOLS` in `neurons/validators/weights.py`).

## Embed tasks

A task is either `crawl` or `embed`. `POST /v1/tasks/lease` takes `{"kind": "embed"}` (crawl when
there is no body) and `POST /v1/validation/lease` takes `{"kinds": ["crawl", "embed"]}`; each kind
has its own queue, its own validation queue, and its own budget, lockout and pool per miner.

The publisher turns every batch of new or changed pages into an input file of the texts the index
embeds: `head` (title and the first 1,500 characters), `full` (title and the first 8,000) and every
passage, cut by `engine/chunking.py`, 500 pages per file. The janitor opens one embed round from the
waiting files, one batch per file, committing to the pages and to the file's `input_sha256`. An
embed task carries the model name, the text count and a presigned link to its input; the miner
uploads one float16 unit vector per `text_id`. Texts are sized to fit the model, so nobody
truncates them.

A validator checks that every text has exactly one vector of the model's size, finite and of unit
length (`vectors_missing`, `vectors_malformed`, `unreadable`), then recomputes 20 texts chosen after
the upload and fails the task if any has a cosine under 0.99 (`vectors_mismatch`). Two hosts of the
same weights agree above 0.9998 and another model lands near 0, so the line sits far from both. A
pass credits the characters of every text, paid out of the embed pool; the API takes the count from
its own round, not from the validator.

The catalog (`embeddings` in `task_api.db`) records, for each page and model, the content hash that
was embedded, the batch and where its vectors are. A page version the current model has already
embedded or queued is not queued again, so an unchanged recrawl costs nothing; a failed batch goes
back out, and a dropped one is marked `dropped` so its pages are queued the next time they change.

## Audits

Each verdict is checked by a second validator with probability `TASK_API_AUDIT_RATE`. The job goes
back to the queue, where the first validator cannot lease it again. If the two agree the verdict
stands, with the lower credit; if they disagree a third validator breaks the tie. A validator whose
verdicts lose more than 30% of at least ten audits can no longer lease validation jobs; `/v1/health`
shows every validator's record. An audit nobody picks up within `TASK_API_AUDIT_WAIT_S` is decided by
the votes so far, and two votes that disagree void the task.

## R2 layout

```
subnet-22/                                                  temp, emptied after a day
  uploads/dt=YYYY-MM-DD/task=<id>/<hotkey>-<lease>.parquet      miner PUT target, removed at /complete
  submitted/dt=YYYY-MM-DD/task=<id>/<hotkey>-<lease>-<n>.parquet  frozen copy that is scored, then published
  embed-inputs/dt=YYYY-MM-DD/<id>.parquet                      texts of one embed batch, removed once published
desearch-pages/                                             permanent
  pages/<domain>/<sha1>                                       latest verified version of one URL, zstd JSON
  changes/dt=YYYY-MM-DD/<time>-<id>.parquet                   every new or changed URL, for indexing and sync
  reports/dt=YYYY-MM-DD/task=<id>.json                        every final verdict, with its samples and votes
  vectors/model=<name>/dt=YYYY-MM-DD/task=<id>.parquet        one embed task: every text with its page, model and vector
```

A page is found from its URL alone:

```python
c = canonicalize(url)
key = page_key(c)                                 # pages/<domain>/<sha1>
doc_id = str(uuid.uuid5(uuid.NAMESPACE_URL, c))   # a stable id for the page
```

Each object holds `url`, `domain`, `title`, `published`, `author`, `lang`, `text`, `fetched_at`,
`content_sha1` and `source="subnet22"`, with `html` left empty, plus what the subnet verified: `doc_id`, `assigned_url`, `final_url`, `canonical`, `status`,
`page_type`, `description`, `json_ld_types`, `headings`, `text_sha256`, `task_id` and `miner`. The
publisher writes only rows for URLs the task was given, skips text that reads as a challenge page,
writes an object only when its content changed, uses conditional writes so concurrent publishers
cannot overwrite each other, and never replaces a newer fetch with an older one. Which fetch is
newer is decided by server time: a row's reported `fetched_at` is clamped to its lease window. Each
write is appended to `changes/` with the full record, so a store can be rebuilt from a few large
files. Change files are named by a sequence number; consumers should re-list a day with a lag and
skip files already read. HTML is not kept past the temp bucket.

A publish job that fails five times is set aside (`publish_set_aside` in `/v1/health`) so it stops
holding back leasing; an upload that expired or changed before publishing is counted in
`publish_lost`. `CF_R2_PAGES_PREFIX` moves the keys under a prefix and is for tests only: consumers
expect `pages/` at the bucket root.

A task still waiting for validation or publishing when its upload expires cannot be published.
Leasing pauses long before that, once `oldest_validation_s` or `oldest_publish_s` in `/v1/health`
passes `TASK_API_MAX_BACKLOG_S`; an expired upload awaiting validation is voided and its URLs are
requeued. When storage fails on the API's side during scoring, the task is handed back without
counting against the miner.

## Tests

```bash
cd task-api && python3 -m pytest -q
```

`pytest.ini` puts the repository root on the path for the shared package. The miner, validator and
shared-package tests run from the root with `pytest`.

The tests use Redis db 14 and an in-memory R2. With `TASK_API_TEST_R2=1` and R2 credentials in
`task-api/.env`, the storage tests run a second time against the real buckets under `_test/`, and
clean up after themselves. Run the suite once at a time: parallel runs share db 14.
