# How Subnet 22 Works

Subnet 22 builds Desearch's web index. Its first task family is crawling: miners fetch web pages,
validators check their work by fetching a sample of the same pages themselves, and the pages that pass
are published to the collection the index is built from. Miners are paid by their share of the
verified pages. Embedding is the next task family; see [Desearch 2.0](./desearch-2.0/README.md).

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

## How miners are paid

Each task family has its own pool of the emission. Every epoch each validator reads the task API's
`/v1/shares`, which gives each miner's share within every pool, and pays each pool out by those
shares. Whatever no pool pays out goes to the subnet's burn hotkey.

| Pool | Part of the emission | A miner's share |
| --- | --- | --- |
| Crawl | 50% | Its fraction of the pages credited in the last 24 hours, if it returned at least 85% of the URLs assigned to it over the same window |
| Burn | the rest | Not a pool: it takes what the pools leave |

Embedding will get its own pool when it opens. If no miner has earned a share in a pool yet, that
pool's part is burned too. If the task API cannot be reached, the validator keeps its last weights.

A miner's **budget** caps how many tasks it can hold from lease to verdict. It starts at 1, grows by
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
| Shared package | Fetching, text extraction, the signed API client and the credit rules, so miners and validators run the same code. | [`desearch/`](../desearch/) |
| Task API | Packs URLs into rounds, leases tasks, collects verdicts, serves shares and the public task logs. | [`task-api/app/`](../task-api/app/) |
| Publisher | Writes verified pages to the permanent pages bucket. | [`task-api/publisher/`](../task-api/publisher/) |
| Round verifier | Checks a closed round's commitment, serve order and signatures. | [`task-api/tools/`](../task-api/tools/) |

The task API's endpoints, round format, audit rules and storage layout are documented in
[task-api/README.md](../task-api/README.md).
