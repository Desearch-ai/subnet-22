# How Subnet 22 Works

Subnet 22 builds Desearch's web index with two task families. In **crawling**, miners fetch web
pages, validators check their work by fetching a sample of the same pages themselves, and the pages
that pass are published to the collection the index is built from. In **embedding**, miners turn
those published pages into vectors, and validators recompute a sample with the same model. Miners
are paid by their share of the verified work in each family.

## The life of a crawl task

1. **Rounds.** The task API packs URLs into batches and commits to them before serving any: it
   publishes the round's manifest hash and picks a seed block ten blocks ahead, whose hash then
   decides the order the batches go out in.
2. **Leases.** A miner leases a task and gets its URLs plus a presigned upload URL. It never holds
   storage credentials.
3. **Crawling.** The miner fetches each URL, extracts the page type, title, dates, headings and main
   text, and uploads one parquet file per task.
4. **Validation.** A validator downloads the upload, checks that every row re-extracts exactly
   from its HTML, and re-fetches a sample of the pages from its own address. It falls back to
   ScrapingDog only for pages its own address cannot load.
5. **Credit.** A passing task is credited row by row at the rate its sample matched. The task API
   computes the credit from the sample outcomes and ignores any credit a validator claims.
6. **Publishing.** The publisher writes each verified page to the permanent pages bucket, only when
   its text changed. The search index is built from that bucket.

## The life of an embed task

1. **Inputs.** Each time the publisher writes new or changed pages, it cuts them into the texts the
   index uses (a head, the full text and every passage) and hands the files to the task API.
2. **Rounds.** The API opens an embed round the same way, committing to each batch's pages and to the
   hash of its input file, and records every page in its catalog for the current model.
3. **Embedding.** A miner leases a task, downloads the input, checks it against the committed hash,
   and uploads one unit vector per text.
4. **Validation.** A validator checks every row is present and well formed, and recomputes a random
   sample of the texts with the same model. Every sample must match.
5. **Credit and publishing.** A passing task is credited with the characters it embedded, and its
   vectors are published next to the pages, marked with the model that made them.
6. **Search.** The engine's sync picks the new vectors up within a minute and the search service
   serves them without a restart; a recrawled page replaces its older copy.

## How miners are paid

Each task family has its own pool of the emission. Every epoch each validator reads the task API's
`/v1/shares`, which gives each miner's share within every pool, and pays each pool out by those
shares. Whatever no pool pays out goes to the subnet's burn hotkey.

| Pool | Part of the emission | A miner's share |
| --- | --- | --- |
| Crawl | 25% | Its fraction of the pages credited in the last 24 hours, if it returned at least 85% of the URLs assigned to it over the same window |
| Embed | 25% | Its fraction of the characters embedded in passing tasks in the last 24 hours |
| Burn | the rest | Not a pool: it takes what the pools leave |

A pool's part never grows with the number of miners in it. If no miner has earned a share in a pool
yet, that pool's part is burned too. If the task API cannot be reached, the validator keeps its last weights.

A miner has a separate **budget** in each pool, capping how many tasks it can hold from lease to verdict. It starts at 1, grows by
one for every task credited for at least 85% of its URLs, and halves on a failed task, an expired
lease or an abandoned task. Two failures the miner caused within 24 hours, at least 5% of its verdicts, lock it out for 12 hours,
and no miner is given a task it held before. The exact failure rules are in
[the task API's scoring section](../task-api/README.md#scoring).

## Why the results can be trusted

- **Committed rounds.** A round's URLs are fixed before its seed block exists, and that block's hash
  decides their order, so the API cannot quietly swap or reorder batches.
- **Signed logs.** Every lease, refusal, completion and hand-back (bar polls over the rate limit) is logged and signed by the task API, and
  `GET /v1/key` publishes the signing key. Anyone can check a closed round with
  `task-api/tools/verify_round.py`.
- **Audits.** A share of verdicts is checked again by a second validator, with a third breaking ties.
  A validator whose verdicts lose more than 30% of at least ten audits can no longer validate.
- **Independent fetches.** Validators fetch pages themselves rather than trusting the miner's copy,
  and text that keeps a page's words but changes its figures still fails.
- **Public verdicts.** Every verdict, with what the validator found for each URL, is public at
  `GET /v1/tasks` and `GET /v1/tasks/{task_id}`.

## Components

| Component | Role | Code |
| --- | --- | --- |
| Miner | Leases tasks, fetches pages through its own proxies, extracts text and uploads it. | [`neurons/miners/`](../neurons/miners/) |
| Validator | Downloads each upload, re-fetches a sample and returns a verdict; sets weights on chain from each pool's shares, burning what no pool pays out. | [`neurons/validators/`](../neurons/validators/) |
| Shared package | Fetching, text extraction, the embedding models and file layouts, and the signed API client, so miners and validators run the same code. | [`desearch/`](../desearch/) |
| Task API | Packs URLs into rounds, leases tasks, collects verdicts, serves shares and the public task logs. | [`task-api/app/`](../task-api/app/) |
| Embed miner | Leases embed tasks and returns one vector per text. | [`neurons/miners/embed.py`](../neurons/miners/embed.py) |
| Publisher | Writes verified pages, the texts to embed and the verified vectors to storage. | [`task-api/publisher/`](../task-api/publisher/) |
| Engine | The search index and API built from the pages and their vectors. | [`engine/`](../engine/) |
| Round verifier | Checks a closed round's commitment, serve order and signatures. | [`task-api/tools/`](../task-api/tools/) |

The task API's endpoints, round format, audit rules and storage layout are documented in
[task-api/README.md](../task-api/README.md).
