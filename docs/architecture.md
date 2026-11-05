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
| Task API | Packs URLs into tasks, hands them to miners, draws which finished uploads validators check, works out what each miner is owed from their reports and the validators' results, and publishes every miner's share. | [`task-api/src/`](../task-api/src/) |
| Miners | Fetch the pages of a task and extract their text, and turn text into vectors on a GPU. | [`neurons/miners/`](../neurons/miners/) |
| Validators | Read each drawn upload from storage, check a sample of its pages against the live page, report pass or fail, and set weights from every miner's logged uploads at the rate their own checks found. | [`neurons/validators/`](../neurons/validators/) |
| Publisher | Writes passed pages and vectors to permanent storage, takes withdrawn pages down, and reports what became of every URL to the bot. | [`publisher/`](../publisher/) |
| Shared package | Fetching, text extraction and embedding formats, so miners and validators run the same code. | [`desearch/`](../desearch/) |

The task API keeps its live queues in Redis and its records (rounds, the signed log, the validators'
results, budgets and what each miner is owed) in SQLite. Files live in two buckets: uploads in
`subnet-22`, a temporary OVHcloud Object Storage bucket whose checked uploads anyone can read at
`https://subnet-22.s3.us-east-va.io.cloud.ovh.us`, and published pages in `desearch-pages`, which keeps the latest version of
every page.

## A crawl task, start to finish

1. **Round.** The task API commits to a round's URL set and to a future block before that block
   exists. Once its hash is known, it cuts the set into tasks of at most 1,000, dealing each site's
   URLs across them in turn so every task has the round's mix, and the same hash decides the order
   they go out in.
2. **Claim.** A miner asks for as many tasks as it can finish and receives, for each, its URLs and an
   upload link. The link accepts one file under one key. The claim lasts 3 minutes plus 10 seconds of
   grace for the upload, counted in finalized chain blocks, and costs nothing if it lapses while the
   API restarts, its storage fails or finality stalls. Miners never hold storage credentials, and a
   miner is never given a task it held before.
3. **Crawl.** The miner fetches every URL, extracts each page's text, and uploads one Parquet file
   with a row per URL, including the page's HTML. When it reports the task
   complete, the task API checks the file is Parquet and within the size limit, and records which
   version of the file was completed. Validators and the publisher read only that version, so a file
   changed after completion fails its check.
4. **Draw.** The API writes a signed note next to the upload (the task's URLs, the file's version and
   the block it was completed at). The hash of the block ten blocks after completion, which nobody
   knew while uploading, decides whether validators check it. Checked are every upload of a new
   hotkey until it has passed 10 checks, the next 10 after a failed check, every upload while a
   hotkey is locked out, and a share of each hotkey's other uploads, about one an hour at least. An
   upload not drawn is finalized at once on the counts the miner reported, and published.
5. **Check.** A drawn upload is listed in `validation/open.json` in the same public bucket. Every
   validator reads that list and the upload from the bucket, and picks the rows to check with the
   same block hash. It then:
   - re-extracts the picked rows from the uploaded HTML, to prove the text came from the page;
   - fetches the same pages itself and compares the text with the miner's;
   - checks a sample of the rows the miner reported as failed.

   It reports the facts of the file (rows, duplicates, rows outside the task, rows per site, rows not
   re-extracted from their HTML) and what it saw for each checked page; reports stay sealed until the
   upload is finalized. A validator whose own fetches failed says so, at no cost to the miner.
   Reporting is the only thing a validator needs the API for.
6. **Finalize.** A checked upload is finalized once every active validator has reported, or at its
   deadline on the reports it has; a validator is active for an hour after its last report. The
   miner's budget, strikes and take-backs follow fixed rules on the facts of the file, which every
   honest validator finds the same; facts two validators disagree on charge nothing. The live-page
   comparison goes into each validator's own weights, and the majority of it decides only what is
   published. A fail on the facts takes back everything the hotkey passed since its last passed
   check, pay and pages, and two among its last 10 cost its last 24 hours of pay and a 48-hour
   lockout. The exact rules are in [Rules](./rules.md).
7. **Publish.** The publisher writes each passed page to the pages bucket, only when its text
   changed and never over a newer fetch, leaving out the rows the checks turned down, and takes
   withdrawn pages down. What became of every URL goes back to the bot
   through the outcome feed, so failed and withdrawn pages are sent again.

## An embed task, start to finish

Embedding is switched off until its inputs and the publishing of vectors below are running; see
[Embedding tasks](./embedding-tasks.md).

1. **Input.** A planner follows the published pages and batches those whose text has no vector yet:
   a page text for each, and passages for long articles. It hands the task API a batch when the API
   has room.
2. **Claim.** A miner running that model receives the batch.
3. **Embed.** The miner runs the model on its GPU and uploads one vector per text; the task API
   refuses a file that does not hold exactly that many.
4. **Check.** A block hash draws a share of each miner's uploads. For each drawn upload, validators
   read a random sample of its vectors and get the same texts embedded again, on their own model
   server or on Desearch's through the task API. Every sample must match. Uploads not drawn are paid
   at the rate the miner's checks paid, and a failed check takes back the miner's uploads since its
   last passed one.
5. **Publish and index.** The publisher checks every vector and writes them, each with its page and
   text, to the pages bucket under `vectors/<model>/` for the index. Vectors of a task taken back are
   withdrawn there and their pages embedded again.

## How the work stays honest

- **Committed rounds.** A round's URL set is fixed before the block that cuts and orders it exists,
  so the task API can neither choose which URLs go together nor reorder them, and every task has the
  round's mix.
- **Blocks, not clocks.** Every deadline and lockout counts finalized chain blocks, so the server's
  clock cannot shorten a claim; outages on the API's side are public and cost nobody.
- **Relative thresholds.** The coverage gate falls with the share of a round validators could load,
  and a site most miners of a round could not load pays and costs nobody that round, so a bad mix of
  URLs hurts no one's standing and a site only one miner can load earns it nothing.
- **Signed logs.** Every claim, refusal and completion is signed by the task API. Anyone can check a
  closed round with [`task-api/tools/verify_round.py`](../task-api/tools/verify_round.py), which reads
  the seed block's hash from a node of their own.
- **One verifiable log.** Every request that changes the accounts, every decision the API makes and every
  block hash it decides on goes into one hash-chained, signed log in the public bucket, and its head is
  committed on chain. Validators check it against their own node and replay it with the API's own rules,
  so a budget, verdict, take-back or share the rules do not give is a fraud proof
  ([Verifiable log](../task-api/README.md#verifiable-log)).
- **Direct to storage.** Uploads go from the miner straight into storage through a link for one
  key, and validators read them from the public bucket. The API never carries the files, so it has
  nothing to alter, and the signed note beside each upload records what the miner was given.
- **Witnessed paths.** Miners can reach the task API through validators' relays, which sign a witness of
  every request they forward; one the API's signed log then lacks is a censorship proof. Upload keys are
  random until the completion, and a storage canary makes any storage outage public and, by rule, harmless.
  See [witness relays](./witness-relays.md).
- **Uploads pinned by version.** On a versioned bucket a completion pins the miner's own version of its
  upload, locked for the check window, and validators read that version, so nobody's later write can fail
  it. Every validator looks at every completed upload and signs what storage held.
- **Unpredictable checks.** Which uploads are checked is decided by a block hash that did not exist
  while the miner uploaded, so no upload is safe to fake.
- **Take-back.** A failed check takes back the miner's unchecked work since its last passed check,
  so faking unchecked uploads costs more than it earns.
- **Validators set their own weights.** Each validator counts every miner's uploads from the
  signed log in the public bucket and pays them at the rate its own checks of that miner found, so
  its weights never come from the task API.
- **No head count decides a penalty.** Budgets, strikes and lockouts follow facts of the file that
  anyone can recompute, and the miner's on-chain incentive, which comes from validators' stake-weighted
  weights. Publishing still follows the validators' majority, and every page published names the
  validators that agreed. A validator is refused only for reports its tasks could not have produced,
  never for disagreeing.
- **Consequences.** Fails on the file's facts and lapsed claims halve a miner's budget, repeated
  strikes lock it out of new tasks for an hour, then 12 hours, then 48, and two such fails among its
  last 10 lock it out for 48 hours.
- **Public results.** Every validator's report, with the rows it paid and what it found for each
  URL, is public at `https://api-22.desearch.ai/v1/tasks`, so anyone can recompute the shares.

## Storage

```
subnet-22 bucket (temporary, objects expire after a day, readable by anyone)
  uploads/          where miners' upload links point, one random key per claim; each completed upload is
                    checked and published from here
  validation/       open.json, the list of uploads waiting for validators
  log/uploads/      every completed upload with the counts its miner reported, signed, for validators to count
  log/journal/      the hash-chained, signed log of every change to the accounts, its head anchored on chain
  outcomes/         what became of every URL, for the bot
  embed-inputs/     texts waiting to be embedded
desearch-pages bucket
  changes/        every new, changed or removed page with its full text, numbered in order, kept ten days
  current/        the latest version of every page, compacted from changes/ into 1,024 shards
  rounds/         each round's URL lists, kept three days
  reports/        every final result with the pages that were checked, kept 30 days
  vectors/        verified vectors, by model
```

How miners earn is explained in [Emission](./emission.md). The task API's endpoints, rules and file
formats are documented in [task-api/README.md](../task-api/README.md).
