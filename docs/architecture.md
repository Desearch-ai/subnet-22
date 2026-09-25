# Architecture

Subnet 22 builds Desearch's web index. A bot finds pages worth keeping, miners crawl them, validators
check the work, and the verified pages are published to the collection the search index is built
from. This page explains how those parts fit together.

```
 Desearch Bot ──new URLs──▶ Task API ◀──leases, uploads──▶ Miners
                               ▲  │
                     verdicts  │  │ jobs to check
                               │  ▼
                           Validators ──sample re-fetch──▶ the web
                               │
 Object storage ◀──publisher───┘ verified work
      │
      ▼
   Engine ──▶ search API
```

## Components

| Component | What it does | Code |
| --- | --- | --- |
| Desearch Bot | Reads the robots.txt and sitemaps of every domain on its list on a schedule, and keeps every URL they list. | [`desearch-bot/`](../desearch-bot/) |
| Feeder | Sends the bot's newest URLs to the task API. | [`task-api/feeder/`](../task-api/feeder/) |
| Task API | Packs URLs into tasks, leases them to miners, hands finished work to validators, turns verdicts into credit, and publishes each miner's share. | [`task-api/app/`](../task-api/app/) |
| Miners | Fetch the pages of a task and extract their text; once embedding opens, turn text into vectors on a GPU. | [`neurons/miners/`](../neurons/miners/) |
| Validators | Check a sample of every task and set weights from the miners' shares. | [`neurons/validators/`](../neurons/validators/) |
| Publisher | Writes verified pages and vectors to permanent storage. | [`task-api/publisher/`](../task-api/publisher/) |
| Engine | Builds the search index from the published pages and vectors, and serves search. | [`engine/`](../engine/) |
| Shared package | Fetching, text extraction and embedding formats, so miners and validators run the same code. | [`desearch/`](../desearch/) |

The task API keeps its live queues in Redis and its records (rounds, the signed log, verdicts,
budgets and credits) in SQLite. Uploads and published data live in object storage (Cloudflare R2).

## A crawl task, start to finish

1. **Round.** The task API packs new URLs into batches, spreading each site across them. It publishes
   a hash of the batches before serving any, and a future block's hash decides the order they go out
   in.
2. **Lease.** A miner asks for a task and receives its URLs and a one-time upload link. Miners never
   hold storage credentials, and a miner is never given a task it held before.
3. **Crawl.** The miner fetches every URL, extracts the title, dates, headings and main text, and
   uploads one Parquet file with a row per URL, including the page's HTML.
4. **Check.** A validator downloads the upload and:
   - re-extracts a sample of rows from the uploaded HTML, to prove the text came from the page;
   - fetches a sample of the pages itself and compares the text with the miner's;
   - checks a sample of the rows the miner reported as failed, to see whether the page really
     cannot be loaded.
5. **Verdict and credit.** The task passes or fails on the rules in
   [the task API's scoring section](../task-api/README.md#scoring). A passing task is credited per row,
   at the rate its sample checked out.
6. **Publish.** The publisher writes each verified page to the pages bucket, only when its text
   changed, and the engine indexes it.

## An embed task, start to finish

Embedding is built and switched off until Desearch's own embedding model ships; see
[Embedding tasks](./embedding-tasks.md).

1. **Input.** When the publisher writes new or changed pages, it cuts them into the texts the index
   uses (a head, the full text and each passage) and hands them to the task API as a batch.
2. **Lease.** A miner receives the batch and the name of the model to run.
3. **Embed.** The miner runs the model on its GPU and uploads one vector per text.
4. **Check.** A validator checks every vector is present and well formed, and recomputes a random
   sample through a hosted service running the same model. Every sample must match.
5. **Publish and index.** The vectors are published next to their pages, and the engine picks them
   up within a minute, without a restart.

## How the work stays honest

- **Committed rounds.** A round's batches are fixed before the block that orders them exists, so the
  task API cannot swap or reorder them.
- **Signed logs.** Every lease, refusal and completion is signed by the task API. Anyone can check a
  closed round with [`task-api/tools/verify_round.py`](../task-api/tools/verify_round.py).
- **Independent checks.** Validators fetch pages themselves rather than trusting the miner's copy.
- **Audits.** A share of verdicts is checked again by a second validator; a validator that loses too
  many of those checks can no longer validate.
- **Consequences.** Failed tasks shrink a miner's budget, and repeated failures lock it out of new
  tasks for 12 hours.
- **Public verdicts.** Every verdict, with what the validator found for each URL, is public at
  `https://task-api.desearch.ai/v1/tasks`.

## Storage

```
temp bucket (short-lived)
  uploads/        miners' uploads, removed once checked
  embed-inputs/   texts waiting to be embedded
pages bucket (permanent)
  pages/          the latest verified version of every page
  changes/        every new or changed page, for the index to follow
  reports/        every final verdict with its samples
  vectors/        verified vectors, by model
```

How miners earn is explained in [Emission](./emission.md). The task API's endpoints, rules and file
formats are documented in [task-api/README.md](../task-api/README.md).
