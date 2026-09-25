# Architecture

Subnet 22 builds Desearch's web index. A bot finds pages worth keeping, miners crawl them, validators
check the work, and the verified pages are published to the collection the search index is built
from. This page explains how those parts fit together.

```
 Desearch Bot ──new URLs──▶ Task API ◀──tasks out──▶ Miners      Validators ──results──▶ Task API
                               │                                      ▲
                               │ one-time upload links                │ reads the open list
                               ▼                                      │ and the uploads
 Miners ──upload──▶ Object storage (public) ──────────────────────────┘──sample re-fetch──▶ the web
                        │
                        │ verified work, through the publisher
                        ▼
                     Engine ──▶ search API
```

## Components

| Component | What it does | Code |
| --- | --- | --- |
| Desearch Bot | Reads the robots.txt and sitemaps of every domain on its list on a schedule, and keeps every URL they list. | [`desearch-bot/`](../desearch-bot/) |
| Feeder | Sends the bot's newest URLs to the task API. | [`task-api/feeder/`](../task-api/feeder/) |
| Task API | Packs URLs into tasks, hands them to miners, hands finished uploads to validators, works out what each miner is owed from the validators' results, and publishes every miner's share. | [`task-api/app/`](../task-api/app/) |
| Miners | Fetch the pages of a task and extract their text; once embedding opens, turn text into vectors on a GPU. | [`neurons/miners/`](../neurons/miners/) |
| Validators | Check a sample of every upload against the live page and set weights from what they found. | [`neurons/validators/`](../neurons/validators/) |
| Publisher | Writes verified pages and vectors to permanent storage. | [`task-api/publisher/`](../task-api/publisher/) |
| Engine | Builds the search index from the published pages and vectors, and serves search. | [`engine/`](../engine/) |
| Shared package | Fetching, text extraction and embedding formats, so miners and validators run the same code. | [`desearch/`](../desearch/) |

The task API keeps its live queues in Redis and its records (rounds, the signed log, the validators'
results, budgets and what each miner is owed) in SQLite. Files live in two Cloudflare R2 buckets:
uploads in `subnet-22`, which is temporary, and published pages in `desearch-pages`, which is
permanent.

## A crawl task, start to finish

1. **Round.** The task API packs new URLs into batches, spreading each site across them. It publishes
   a hash of the batches before serving any, and a future block's hash decides the order they go out
   in.
2. **Claim.** A miner asks for a task and receives its URLs and an upload link. The link points into
   object storage, accepts one file under one key, and expires with the claim after 15 minutes.
   Miners never hold storage credentials, and a miner is never given a task it held before.
3. **Crawl.** The miner fetches every URL, extracts the title, dates, headings and main text, and
   writes one Parquet file with a row per URL, including the page's HTML, straight to storage
   through that link. When the miner reports the task complete, the task API checks the file's
   size, copies it to a key only the API can write, and removes the original. That frozen copy is
   what gets checked: it is exactly what the miner uploaded, and neither side can change it afterwards.
4. **Check.** Every validator checks every upload, and finds them in storage rather than by asking
   the API. When the API freezes an upload it writes a signed note next to it, saying which URLs
   the miner was given and at which block the upload was frozen, and it keeps a list of the open
   uploads at a fixed key in the same public bucket. Each validator reads that list, checks the
   API's signature on every note, and reads the frozen copy straight from the bucket. The rows to
   check are picked with a seed nobody knew when the upload was frozen: the hash of the chain block
   ten blocks after the freeze, which the validator takes from the chain itself. It then:
   - re-extracts the picked rows from the uploaded HTML, to prove the text came from the page;
   - fetches the same pages itself and compares the text with the miner's;
   - checks a sample of the rows the miner reported as failed, to see whether the page really
     cannot be loaded.
   Each validator then reports pass or fail to the API, with what it saw for every page it
   checked. A validator whose own fetches failed reports that instead, and it costs the miner
   nothing. Reports stay sealed until the upload is finalized. Reporting is the only thing a
   validator needs the API for while checking; it also reads the API's coverage figures once per
   epoch before setting weights.
5. **Finalize.** An upload is finalized once every active validator has reported, or at its deadline
   with more than half of them. A validator is active for an hour after any report it sends, even
   one that arrives too late to count, so a quick report never finalizes an upload that a slower
   validator is still checking. The majority
   decides pass or fail. A passing upload pays the miner for its rows at the rate the checked pages
   matched: 35 of 40 rows when 7 of 8 checked pages matched. When the validators in the majority
   arrived at different counts, the lower middle value is paid. A validator that disagreed with the
   majority is marked. With one active validator, its report alone finalizes the upload; below the
   quorum at the deadline, nothing is decided and the task goes back out at no cost to the miner.
   The exact rules are in [the task API's scoring section](../task-api/README.md#scoring).
   Everything an upload changes, from what the miner is owed to its coverage to the validators'
   standing, is written in one transaction.
6. **Publish.** The publisher writes each verified page to the pages bucket, only when its text
   changed and never over a newer fetch, leaving out the rows the checks turned down, and the
   engine indexes it.

## An embed task, start to finish

Embedding is built and switched off until Desearch's own embedding model ships; see
[Embedding tasks](./embedding-tasks.md).

1. **Input.** When the publisher writes new or changed pages, it cuts them into the texts the index
   uses (a head, the full text and each passage) and hands them to the task API as a batch.
2. **Claim.** A miner receives the batch and the name of the model to run.
3. **Embed.** The miner runs the model on its GPU and uploads one vector per text.
4. **Check.** A validator checks every vector is present and well formed, and recomputes a random
   sample through a hosted service running the same model. Every sample must match.
5. **Publish and index.** The vectors are published next to their pages, and the engine picks them
   up within a minute, without a restart.

## How the work stays honest

- **Committed rounds.** A round's batches are fixed before the block that orders them exists, so the
  task API cannot swap or reorder them.
- **Signed logs.** Every claim, refusal and completion is signed by the task API. Anyone can check a
  closed round with [`task-api/tools/verify_round.py`](../task-api/tools/verify_round.py).
- **Direct to storage.** Uploads travel from the miner into object storage through a link the task
  API issues for one key and one claim, and validators read them from the public bucket without
  asking anyone. The API checks what landed and freezes it; it never carries the files, so there
  is nothing for it to alter, and the signed note next to each upload lets anyone check later what
  the miner was given.
- **No single validator publishes.** An upload reaches the corpus only after it is finalized, which
  takes more than half of the active validators. Every published page names the validators that
  agreed on it.
- **Validators are judged by their own work.** Each validator sets weights from the results it
  reached itself, so a validator whose results stray from the others' stands out in consensus and
  earns less. One that cannot check uploads sets no weights, and one that keeps disagreeing with
  the majority stops receiving uploads. Two validators that both pass an upload but differ by more
  than 15% on how many rows to pay count as disagreeing.
- **Consequences.** Failed tasks shrink a miner's budget, and repeated failures lock it out of new
  tasks for 12 hours.
- **Public results.** Every validator's report, with the rows it paid and what it found for each
  URL, is public at `https://task-api.desearch.ai/v1/tasks`, so anyone can recompute the shares.

## Storage

```
subnet-22 bucket (temporary, objects expire after a day, readable by anyone)
  uploads/          where miners' upload links point, removed once the task is complete
  submitted/        the frozen copies validators check, each with its signed note beside it
  validation/       open.json, the list of uploads waiting for validators
  embed-inputs/     texts waiting to be embedded
desearch-pages bucket (permanent)
  pages/          the latest verified version of every page
  changes/        every new or changed page, for the index to follow
  reports/        every final result with the pages that were checked
  vectors/        verified vectors, by model
```

How miners earn is explained in [Emission](./emission.md). The task API's endpoints, rules and file
formats are documented in [task-api/README.md](../task-api/README.md).
