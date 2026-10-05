# Architecture

Subnet 22 builds Desearch's web index. A bot finds pages worth keeping, miners crawl them, validators
check a drawn share of the work, and the pages are published to the collection the search index is
built from. This page explains how those parts fit together.

```
 Desearch Bot ──new URLs──▶ Task API ──tasks and upload links──▶ Miners
                               │  ▲                                │
              list of uploads  │  │ pass or fail                   │ upload
              waiting to be    │  │                                ▼
              checked          │  └──────── Validators ◀──read─── Storage (public bucket)
                               └───────────────▶│                       │
                                                │ re-fetch a sample      │ verified pages,
                                                ▼                        │ through the publisher
                                             the web                     ▼
                                                                    Engine ──▶ search API
```

## Components

| Component | What it does | Code |
| --- | --- | --- |
| Desearch Bot | Reads the robots.txt and sitemaps of every domain on its list on a schedule, keeps every URL they list, and fills the task API's queue with new, changed and retried pages as it has room. | [`desearch-bot/`](../desearch-bot/) |
| Task API | Packs URLs into tasks, hands them to miners, draws which finished uploads validators check, works out what each miner is owed from their reports and the validators' results, and publishes every miner's share. | [`task-api/app/`](../task-api/app/) |
| Miners | Fetch the pages of a task and extract their text; once embedding opens, turn text into vectors on a GPU. | [`neurons/miners/`](../neurons/miners/) |
| Validators | Read each drawn upload from storage, check a sample of its pages against the live page, report pass or fail, and set weights from every miner's logged uploads at the rate their own checks found. | [`neurons/validators/`](../neurons/validators/) |
| Publisher | Writes passed pages and vectors to permanent storage, takes withdrawn pages down, and reports what became of every URL to the bot. | [`task-api/publisher/`](../task-api/publisher/) |
| Engine | Builds the search index from the published pages and vectors, and serves search. | [`engine/`](../engine/) |
| Shared package | Fetching, text extraction and embedding formats, so miners and validators run the same code. | [`desearch/`](../desearch/) |

The task API keeps its live queues in Redis and its records (rounds, the signed log, the validators'
results, budgets and what each miner is owed) in SQLite. Files live in two Cloudflare R2 buckets:
uploads in `subnet-22`, which is temporary and readable by anyone at `https://r2.desearch.ai`, and
published pages in `desearch-pages`, which is permanent.

## A crawl task, start to finish

1. **Round.** The task API packs new URLs into tasks of 1,000, spreading each site across them. It
   publishes a hash of the batches before serving any, and a future block's hash decides the order
   they go out in.
2. **Claim.** A miner asks for as many tasks as it can finish and receives, for each, its URLs and an
   upload link. The link accepts one file under one key and lasts as long as the claim: 3 minutes,
   plus 10 seconds of grace for the upload. Miners never hold storage credentials, and a miner is
   never given a task it held before.
3. **Crawl.** The miner fetches every URL, extracts the title, dates, headings and main text, and
   uploads one Parquet file with a row per URL, including the page's HTML. When it reports the task
   complete, the task API checks the file is Parquet and within the size limit, copies it to a key
   only the API can write, and removes the original. Validators check that frozen copy.
4. **Draw.** The API writes a signed note next to the frozen copy (the task's URLs and the block it
   was frozen at). The hash of the block ten blocks after the freeze, which nobody knew while
   uploading, decides whether validators check it. Checked are every upload of a new hotkey until
   it has passed 10 checks, the next 10 after a failed check, every upload while a hotkey is locked
   out, and a share of each hotkey's other uploads, about one an hour at least. An upload not drawn
   is finalized at once on the counts the miner reported, and published.
5. **Check.** A drawn upload is listed in `validation/open.json` in the same public bucket. Every
   validator reads that list and the upload from the bucket, and picks the rows to check with the
   same block hash. It then:
   - re-extracts the picked rows from the uploaded HTML, to prove the text came from the page;
   - fetches the same pages itself and compares the text with the miner's;
   - checks a sample of the rows the miner reported as failed.

   It reports pass or fail to the API, with what it saw for each checked page; reports stay sealed
   until the upload is finalized. A validator whose own fetches failed says so, at no cost to the
   miner. Reporting is the only thing a validator needs the API for.
6. **Finalize.** A checked upload is finalized once every active validator has reported, or at its
   deadline on the reports it has; a validator is active for an hour after its last report. The
   majority decides pass or fail, and a pass pays the miner's rows at the rate the checked pages
   matched (35 of 40 rows when 7 of 8 matched). One active validator finalizes alone. A failed
   check takes back everything the hotkey passed since its last passed check, pay and pages, and
   two failed checks among its last 10 cost its last 24 hours of pay and a 48-hour lockout. The
   exact rules are in [the task API's scoring section](../task-api/README.md#scoring).
7. **Publish.** The publisher writes each passed page to the pages bucket, only when its text
   changed and never over a newer fetch, leaving out the rows the checks turned down, takes
   withdrawn pages down, and the engine indexes them. What became of every URL goes back to the bot
   through the outcome feed, so failed and withdrawn pages are sent again.

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
- **Direct to storage.** Uploads go from the miner straight into storage through a link for one
  key, and validators read them from the public bucket. The API never carries the files, so it has
  nothing to alter, and the signed note beside each upload records what the miner was given.
- **Unpredictable checks.** Which uploads are checked is decided by a block hash that did not exist
  while the miner uploaded, so no upload is safe to fake.
- **Take-back.** A failed check takes back the miner's unchecked work since its last passed check,
  so faking unchecked uploads costs more than it earns.
- **Validators set their own weights.** Each validator counts every miner's uploads from the
  signed log in the public bucket and pays them at the rate its own checks of that miner found, so
  its weights never come from the task API.
- **Validators decide checked uploads.** A checked upload is decided by the active validators, and
  every page published from it names the validators that agreed. One that keeps disagreeing with
  the majority stops receiving uploads.
- **Consequences.** Failures and lapsed claims halve a miner's budget, repeated strikes lock it out
  of new tasks for an hour, then 12 hours, then 48, and two failed checks among its last 10 lock it
  out for 48 hours.
- **Public results.** Every validator's report, with the rows it paid and what it found for each
  URL, is public at `https://api-22.desearch.ai/v1/tasks`, so anyone can recompute the shares.

## Storage

```
subnet-22 bucket (temporary, objects expire after a day, readable by anyone)
  uploads/          where miners' upload links point, removed once the task is complete
  submitted/        the frozen copies validators check, each with its signed note beside it
  validation/       open.json, the list of uploads waiting for validators
  log/uploads/      every completed upload with the counts its miner reported, signed, for validators to count
  outcomes/         what became of every URL, for the bot
  embed-inputs/     texts waiting to be embedded
desearch-pages bucket (permanent)
  pages/          the latest verified version of every page
  changes/        every new, changed or removed page, for the index to follow
  reports/        every final result with the pages that were checked
  vectors/        verified vectors, by model
```

How miners earn is explained in [Emission](./emission.md). The task API's endpoints, rules and file
formats are documented in [task-api/README.md](../task-api/README.md).
