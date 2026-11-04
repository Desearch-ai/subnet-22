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
| `tools/check_witnesses.py` | turns requests relaying validators witnessed, missing from the signed log, into censorship proofs |

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
upload link for one key; it never holds storage credentials. The key is 128 random bits
(`uploads/<32 hex>.parquet`) and names no miner, task or claim; it reaches the public log only with the
completion. The claim's time starts when the API hands the tasks over. On completion the API records the
upload's entity tag in the signed note. On a versioned bucket it pins the version the miner's own upload
made (the completion names it, from the upload's answer) and records it in the note, and validators read
that version by id, so nothing written to the key afterwards, by anyone, changes what they check. Without
versions, validators and the publisher read the upload with `If-Match` on its tag, so a file changed
afterwards is answered with 412 and fails its check (`upload_changed`); a completion naming an `etag` that
the key no longer holds is refused with 409. See [witness relays](../docs/witness-relays.md). A completion counts as on time when it
reaches the API inside the claim, however long the API then takes over it; a file written after the
completion reached the API is refused. Sending the same completion again returns the first one's
answer. A miner is never given a batch it held before.

**Which uploads are checked.** On completion the API writes a signed note beside the upload (the
task's URLs, its entity tag and the completion block). Once the block ten blocks after that exists,
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

**Signed log.** Every claim, refusal and completion is signed by the API, and names the hash of the
miner's signed request it answers (`request`), so a validator that relayed the request can show whether the
log holds it. `tools/verify_round.py` checks a closed round end to end, reading the seed block's hash from
your own node, and `tools/check_witnesses.py` turns relayed requests the log lacks into censorship proofs.
Beyond that, every change to the accounts goes into one hash-chained log that is published, anchored on
chain and replayed by validators; see [Verifiable log](#verifiable-log).

**Storage canary.** Every minute the API writes an object shaped like a miner's upload through a presigned
link and reads it back through another; failures are public at `/v1/storage/probes` and every probe is a
`probe` entry in the log. With `TASK_API_STORAGE_OUTAGE_FORGIVES=1` a claim that lapses while a probe failed
costs its miner nothing; the lapse's entry says which copy decided, when the claim began and why it was
forgiven, and a replay decides it again from the logged probes.

## Verifiable log

The API is run as a state machine anyone can check. Every input that changes the accounts, and what the
API made of it, is one entry in a log, written in the same SQLite transaction as the change:

```json
{"v": 1, "seq": 812, "prev": "<sha256 of entry 811>", "at": 1793781693.42, "kind": "finalize", "input": {...}, "output": {...}}
```

An entry's hash is the SHA-256 of its canonical JSON (`json.dumps(sort_keys=True, separators=(",", ":"),
ensure_ascii=False)`), and every entry names the hash of the one before, so no entry can be altered,
dropped, reordered or forked without every later hash changing.

| Kind | Input | Output |
| --- | --- | --- |
| `start` | the version, the signer and the knobs that shape decisions | — |
| `snapshot` | when | the digest of the accounts, and the accounts in a blob named by it (one at the first start, then every `TASK_API_SNAPSHOT_EVERY_S`) |
| `requests` | the hash of every authenticated request since the last, and the block | — |
| `claim`, `complete`, `abandon`, `report`, `release`, `enqueue` | the signed request: method, path, hotkey, timestamp, nonce, signature and body; a completion's Object Lock (version, until, whether it held) | the receipts (each naming the request's hash), the job, the vote, the budget, the round's commitment, or why it was refused |
| `miner`, `lapse`, `settle`, `finalize`, `take_back`, `recheck` | what the rule read: the job (with its pinned version), the voters in order, the seed block and its hash, the upload counters, the claim's start and expiry | the budget, whether the upload is checked, the verdict, its credit, the uploads taken back, why a lapse was forgiven (`restart`, `storage_outage`) |
| `probe` | the storage canary's probe: when, the block, whether it passed and the step that failed | — |
| `reveal`, `close`, `embed_round` | the round, its seed block and that block's hash | the seed, the serve order, the round log's Merkle root |
| `metagraph` | the neurons and their stakes, when who may vote changed | the validators |
| `prune`, `shares`, `anchor`, `receipt` | | the shares as served, an anchor the chain confirmed, receipts signed elsewhere |

A request body over 2 kB, a job's URLs, a manifest and a snapshot are kept in blobs the entry names by
their SHA-256 (`{"blob": "<sha256>", "bytes": n}`); a job's URL blob is named by the `urls_hash` its round
committed to. Presigned upload links are never logged, and an upload's random key enters the log only with
its completion.

**Files.** Every `TASK_API_JOURNAL_EVERY_S` the janitor writes the entries not yet published into
`log/journal/seq/<n>.json` in the uploads bucket, after the blobs they name (`log/journal/blobs/<sha256>`),
and points `log/journal/latest.json` at it. Each file holds `first`, `last`, `prev` (the hash before its
first entry), `head` (its last entry's hash), the entries, the signer and a signature over the rest. Two
files with one number and different heads, both signed, prove a fork on their own.

**Anchors.** With `TASK_API_ANCHOR_NETWORK` set, the API commits the head to the chain with
`Commitments.set_commitment(netuid, [Raw("sn22j1:<seq>"), Sha256(<hash of entry seq>)])`, signed by
`TASK_API_ANCHOR_KEY_URI` (the log's key unless set). Commitments are free, live in chain state
(`Commitments.CommitmentOf(netuid, hotkey)`) where any node answers, and a hotkey may make 31 per epoch
(3,100 bytes, each call counted as 100), so the API anchors every `TASK_API_ANCHOR_EVERY_BLOCKS` and, once a
round opened, after `TASK_API_ANCHOR_MIN_BLOCKS`, at most `TASK_API_ANCHOR_PER_EPOCH` per 361 blocks. A
`System.remark` would need no registration but a funded account and an indexer to find it again. The
anchoring hotkey must be registered on the subnet (and kept from being pruned, e.g. as an owner-immune
uid) and hold the existential deposit. Each confirmed anchor is logged as an `anchor` entry.

**Replay.** [`src/replay.rs`](src/replay.rs) rebuilds the accounts from the log with the API's own rule
functions on an SQLite of its own, from the first `snapshot`:
- every logged request must be signed by the hotkey it names, and every blob must hash to its name;
- a validator's vote is built again from its signed report; a verdict is decided again from the votes of
  the voters the entry names, all of which must have been reported and none left out;
- whether a seeded upload is checked is drawn again from the seed block's hash, the counters and the
  re-check state; budgets, strikes, lockouts, take-backs and full penalties follow from the same rules;
- shares and every later snapshot's digest must equal the replay's own.

A difference the API decided inside one transaction is `fraud`. A check the API makes on reads outside the
transaction that logs it (a claim against the budget or a lockout) is only `suspect`. The order of entries,
the times they state and the Redis counters they quote are the API's to choose and are taken as given. The
replay ships as `task-api-replay --dir <folder> --state <file> --signer <ss58> [--join]`, reading the
bucket's layout (`seq/`, `blobs/`) from a folder and going on from its saved state; it prints each
difference as a JSON line.

**Signed reads.** Every answered `GET` carries `X-Desearch-Signer`, `X-Desearch-Signed-At` and
`X-Desearch-Signature`, a signature over the canonical JSON
`{"at": <X-Desearch-Signed-At>, "path": "<path and query>", "sha256": "<SHA-256 of the body>"}`, so a
page or a screenshot of one can be shown to come from the API. The UI reads these endpoints; it does not
check the signatures itself yet.

**Taken as given until blocks A and D.** Times are the API's clock (A moves them to block heights). Omitted
requests are caught by the witness relays: a relay compares its signed requests with the `requests` entries
of the anchored log (`JournalLogView`), falling back to the signed receipts. The key is whoever holds `TASK_API_KEY_URI` (validators pin it in
[`desearch/trust.py`](../desearch/trust.py) until D binds it to an attested enclave).

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

Embed tasks are off until `TASK_API_EMBED_TASKS=1`. The flow:

1. **Planner** (`publisher plan`): follows the change log, keeps English pages of v2 extraction with
   enough text whose text has no vector yet, and writes their texts to input files in the temporary
   bucket (`embed/inputs/`), up to 1,000 pages and 3 million characters a file. Every page gives a page
   text (title, description, the start of its text); a page over 2,000 characters also gives up to 32
   passages split at its markdown headings, never inside a table row, a list item or a code block, each
   starting with the title and the section's heading. With `PLANNER_PAGE_FIRST=1`, pages outside the
   fresh lane give their page text alone. It hands an input to the API (`embed:inputs` in Redis) only
   while `GET /v1/room?kind=embed` has room, and plans at most `PLANNER_PAGES_PER_HOUR` pages an hour.
2. **Task API:** each input becomes one task naming the model (`TASK_API_EMBED_MODEL`). A claim must
   list that model in `models` or is refused with `MODEL_MISMATCH`, at no cost. The miner uploads one
   vector per text, as raw float16 in the input's order, within `TASK_API_EMBED_CLAIM_TTL`; a
   completion whose file is not exactly texts × dimensions × 2 bytes is refused.
3. **Check picking:** like crawl uploads, by the seed: every embed upload of a hotkey with fewer than
   10 passed embed checks, the next 10 after a failed one, otherwise a share
   (`TASK_API_EMBED_CHECK_SHARE`) within `TASK_API_EMBED_CHECKS_PER_HOUR`. An upload not picked is
   finalized as passed and paid its characters. A failed check takes back the miner's embed uploads
   since its last passed embed check, and two among its last 10 bring the full penalty, in the embed
   pool only.
4. **Validators** read the 48 sampled vectors of a picked upload by byte range and get the same texts
   embedded again with the same model, each of which must reach a cosine similarity of 0.999. A
   validator with its own model server uses it; the others ask `POST /v1/embed/{task_id}/reference`
   with the sampled text ids (at most 64 of one upload), and the API reads those texts from the
   upload's input, embeds each once on Desearch's model servers (`TASK_API_EMBED_REFERENCE_URL`) and
   keeps it for the next validator, which waits while it is computed. A picked embed upload waits only
   for validators that vote on embed uploads; one nobody scored by its deadline is void, deleted and
   handed to another miner, at no cost. A pass pays the characters embedded. Embed uploads awaiting a
   verdict count against the planner's room. A task that fails `TASK_API_MAX_ATTEMPTS` times is
   dropped and its pages go back to the planner (`embed:dropped`), which plans them again the next
   time the log shows them. A task is also dropped when it is next claimed, at no cost to that
   miner, once its input is older than `TASK_API_EMBED_INPUT_MAX_AGE_S` (inputs are kept a day) or
   miners have handed it back `TASK_API_MAX_HANDBACKS` times by abandoning it or letting the claim
   expire; the miner draws another task instead.
5. **Publisher:** writes the vectors of every finalized embed task that passed or was not picked, each
   with its page key, URL, text hash, kind and index, to `vectors/<model>/log/` in the pages bucket,
   numbered under `vectors/<model>/seq/`, and deletes the upload and the input. A vector that is not
   finite or not of length 1 is left out, its page goes back to the planner, and the miner's next 10
   uploads are checked (`embed:malformed`). A task taken back is withdrawn from the vectors feed by a
   withdrawal entry ([publisher README](../publisher/README.md#vectors)) and its pages go back to the
   planner.

The miner-facing description is [Embedding tasks](../docs/embedding-tasks.md).

The planner keeps its place in the log and the version of every page it planned in RocksDB under
`PLANNER_DIR`, reads the same bucket settings as the publisher, and is configured with:

| Variable | Default | |
| --- | --- | --- |
| `TASK_API_URL` | `https://api-22.desearch.ai` | where it asks for room |
| `TASK_API_REDIS` | `redis://localhost:6379/15` | where it hands inputs over |
| `PLANNER_DIR` | `/home/publisher/planner` | its state |
| `PLANNER_PAGES_PER_HOUR` | 100000 | pages planned an hour at most, and with the fresh lane's handed over an hour at most, from the start; the log's pages past it wait and are planned later |
| `PLANNER_PAGES_PER_INPUT` | 1000 | pages per input file at most |
| `PLANNER_CHARS_PER_INPUT` | 3000000 | characters of text per input file at most |
| `PLANNER_FLUSH_S` | 600 | a part-filled input is written once its first page waited this long |
| `PLANNER_MAX_PENDING` | 50 | input files not yet handed over; past this, pages wait and are planned later |
| `PLANNER_READS` | 16 | change files fetched ahead while the planner plans, from the log and for waiting pages; they are planned in log order |
| `PLANNER_WRITES` | 8 | input files written to the temporary bucket at once; they are saved in the order cut and handed over fresh lane first, then oldest first |
| `PLANNER_PENDING_MAX_AGE_S` | 43200 | an input waiting longer is dropped before the temporary bucket deletes its file; its pages are planned again when the log shows them next |
| `PLANNER_MIN_CHARS` | 300 | characters of text a page needs |
| `PLANNER_LANGS` | `en` | languages planned |
| `PLANNER_EXCLUDE_FILE` | — | a file of domains, one a line, never planned, with their subdomains |
| `PLANNER_LIST_FILE` | — | the bot's crawl list (`domains.json`); when set, only its sites and the hosts under them are planned, each up to a third of its crawl cap and a news site without a cap, and the file is read again when it changes |
| `PLANNER_START_AFTER` | the newest change file | where a planner without state starts |

## Run

The API is built from `Dockerfile.rust` at the repository root. It keeps its SQLite file in `/data`
and needs a Redis (`TASK_API_REDIS`); the [publisher](../publisher/) runs on its own machine:

```bash
docker build -f task-api/Dockerfile.rust -t task-api .
docker run -d --env-file task-api/deploy/.env.example -e TASK_API_REGISTRY=chain -e TASK_API_SEEDS=chain -v task-api-data:/data -p 127.0.0.1:8080:8080 task-api
curl -s localhost:8080/v1/ping
```

Fill in the storage credentials, admin hotkeys, signing key and Redis URL in the env file first. Behind a
proxy that checks `/v1/ping`, a deploy can roll: the new copy serves next to the old one until the old
one stops, and only one copy at a time runs the background work.

**Database size.** Once a round's tasks are queued, SQLite keeps only each batch's count, hash and
input; the URLs and an embed input's page list live in the task payloads. A round is deleted a week after
it closes. SQLite reuses the space this frees, but the file never shrinks. To shrink it, stop the API
(no copy may run) and write a compact copy next to it, on the host that holds the volume:

```bash
set -e
cd /var/lib/docker/volumes/task-api-data/_data
sqlite3 task_api.db "PRAGMA wal_checkpoint(TRUNCATE)"
sqlite3 task_api.db "PRAGMA cache_size = -4000000; VACUUM INTO 'task_api.compact.db'"
test "$(sqlite3 task_api.compact.db 'PRAGMA quick_check')" = ok
sqlite3 task_api.compact.db "PRAGMA journal_mode = WAL"
mkdir before-compact
mv task_api.db before-compact/
mv task_api.db-wal task_api.db-shm before-compact/ 2>/dev/null || true
mv task_api.compact.db task_api.db
chown 10001:10001 task_api.db*
```

Then start the API. To go back, stop it, move the files in `before-compact/` back and `chown` them the
same way; delete them once the API serves. The copy needs free disk for the live data, not the whole
file, and is written at a few hundred MB a second.

It needs two buckets:

- `subnet-22`, the temporary bucket for uploads, with a lifecycle rule that deletes objects after one day.
  `CF_R2_ENDPOINT`, `CF_R2_REGION` (`auto` by default, as R2 signs), `CF_R2_ACCESS_KEY_ID` and
  `CF_R2_SECRET_ACCESS_KEY` reach it;
- `desearch-pages`, the bucket for published work, with lifecycle rules that delete `changes/` after ten
  days, `rounds/` after three days and `reports/` after 30 days; `current/` is kept by the compactor. It is reached with the uploads bucket's credentials unless
  `PAGES_S3_ENDPOINT`, `PAGES_S3_REGION`, `PAGES_S3_ACCESS_KEY_ID` and `PAGES_S3_SECRET_ACCESS_KEY`
  (and `PAGES_S3_BUCKET` when the name differs) put it on another S3 store; the publisher and the
  compactor read the same settings.

For a versioned bucket under Object Lock, so that no overwrite can fail a miner, see
[witness relays → bucket configuration](../docs/witness-relays.md#bucket-configuration).

Validators read `subnet-22` without credentials. On a store that keeps buckets private and opens
objects one by one, as OVHcloud Object Storage does, set `CF_R2_PUBLIC_OBJECTS=1` for the API and the
publisher: the API writes its own files readable by anyone and opens each upload once its completion
is checked, and the planner writes embed inputs the same way. On R2, serve the bucket on a custom
domain instead, not the rate-limited `r2.dev` URL. Objects are readable but not listable, which is
why the API keeps `validation/open.json`. Mainnet serves it at
`https://subnet-22.s3.us-east-va.io.cloud.ovh.us`, the validator's default `--neuron.storage_url`.

The [bot](../desearch-bot/README.md) fills the queue with the admin hotkey (`--task-api`).

| Variable | Default | |
| --- | --- | --- |
| `TASK_API_REGISTRY` | — | `chain` or `local` |
| `TASK_API_SEEDS` | `local` | `chain` or `local`: where round seeds come from |
| `TASK_API_NETWORK` | `finney` | the chain in `chain` mode: a network name or JSON-RPC URL, or several comma-separated to fall back on in order |
| `TASK_API_REDIS` | `redis://localhost:6379/15` | the Redis URL |
| `TASK_API_LISTEN` | `0.0.0.0:8080` | the address the API listens on |
| `TASK_API_ADMIN_HOTKEYS` | — | hotkeys allowed to enqueue |
| `TASK_API_KEY_URI` | — | the key the log is signed with; required in `chain` mode. Validators pin its address in [`desearch/trust.py`](../desearch/trust.py), so a new key needs a release |
| `TASK_API_JOURNAL_EVERY_S` | 30 | seconds between published log files |
| `TASK_API_JOURNAL_SHARES_EVERY_S` | 600 | seconds between the shares written into the log |
| `TASK_API_SNAPSHOT_EVERY_S` | 21600 | seconds between snapshots of the accounts in the log |
| `TASK_API_JOURNAL_KEEP_S` | 691200 | seconds published log entries stay in SQLite |
| `TASK_API_ANCHOR_NETWORK` | — | the chain the log's head is anchored on (a network name or URL); no anchors without it |
| `TASK_API_ANCHOR_NETUID` | `TASK_API_NETUID` | the subnet the anchoring hotkey is registered on |
| `TASK_API_ANCHOR_KEY_URI` | `TASK_API_KEY_URI` | the hotkey that commits the anchors |
| `TASK_API_ANCHOR_EVERY_BLOCKS` | 12 | blocks between anchors |
| `TASK_API_ANCHOR_MIN_BLOCKS` | 3 | blocks between anchors at least, once a round opened |
| `TASK_API_ANCHOR_PER_EPOCH` | 30 | anchors per 361 blocks at most; the chain allows 31 |
| `TASK_API_CLAIM_TTL` | 180 | seconds a miner is given to upload a task after claiming it; the claim and its upload link last 10 seconds longer |
| `TASK_API_POLL_RATE` | 0.5 | claim requests per second one hotkey may make |
| `TASK_API_VALIDATION_TTL` | 900 | seconds validators have to report once the upload is open |
| `TASK_API_ACTIVE_S` | 3600 | seconds since its last report a validator counts as active |
| `TASK_API_LEDGER_DELAY_S` | 0 | seconds finalized uploads and shares stay out of public view |
| `TASK_API_MAX_ATTEMPTS` | 3 | times a task is retried before it is dropped |
| `TASK_API_CHECK_SHARE` | `SHARE` in [`app/sampling.py`](app/sampling.py) | share of an established hotkey's uploads drawn for a check; 1 checks every upload |
| `TASK_API_QUEUE_TARGET` | 1200 | tasks the bot keeps queued and waiting for their round's reveal |
| `TASK_API_RELEASE_SPREAD_S` | 60 | seconds a revealed round's tasks are released over, one by one in the round's order; 0 releases them all at once |
| `TASK_API_READS_PER_MINUTE` | 120 | `GET` requests one IP may make in a minute, outside the log endpoints |
| `TASK_API_LOG_READS_PER_MINUTE` | 60 | log requests (tasks, votes, miners, validators, overview) one IP may make in a minute |
| `TASK_API_FAILED_WRITES_PER_MINUTE` | 30 | failed sign-ins after which one IP's writes are refused for the rest of the minute |
| `TASK_API_CORS_ORIGINS` | local dev servers | comma-separated origins whose pages may read the API from a browser |
| `TASK_API_EMBED_TASKS` | 0 | 1 opens embed tasks |
| `TASK_API_EMBED_MODEL` | `desearch-4b-v1` | the model embed tasks name, from [`desearch/embedding.py`](../desearch/embedding.py) |
| `TASK_API_EMBED_CLAIM_TTL` | 300 | seconds a miner is given to upload an embed task after claiming it |
| `TASK_API_EMBED_QUEUE_TARGET` | 100 | embed tasks the planner keeps queued and waiting for their round's reveal |
| `TASK_API_EMBED_INPUT_MAX_AGE_S` | 72000 | seconds after the planner wrote an input that its task is dropped at its next claim |
| `TASK_API_MAX_HANDBACKS` | 5 | hand-backs (abandons and expired claims) after which an embed task is dropped at its next claim |
| `TASK_API_EMBED_CHECK_SHARE` | 0.05 | share of an established hotkey's embed uploads drawn for a check; 1 checks every upload |
| `TASK_API_EMBED_CHECKS_PER_HOUR` | 600 | drawn embed checks an hour across all miners; the share is lowered to stay within it |
| `TASK_API_EMBED_REFERENCE_URL` | — | Desearch's model servers (each an OpenAI-compatible `/v1/embeddings` serving the model under its name) for validators without their own, comma-separated and tried in order; without one, the reference endpoint answers 503 |
| `TASK_API_EMBED_REFERENCE_KEY` | — | those servers' key |
| `TASK_API_EMBED_REFERENCE_CALLS` | 4 | requests to those servers at once |
| `TASK_API_CANARY_EVERY_S` | 60 | seconds between storage canary probes; 0 turns the canary off |
| `TASK_API_STORAGE_OUTAGE_FORGIVES` | 0 | 1: a claim that lapses while a canary probe failed costs nothing (cause `storage_outage`) |
| `TASK_API_UPLOAD_LOCK_S` | 0 | seconds each pinned upload version is put under Object Lock in compliance mode; needs a versioned bucket with Object Lock on |

While the Redis key `reference:servers` holds a comma-separated list of servers, it is used instead of
`TASK_API_EMBED_REFERENCE_URL`, so servers are added or replaced without a restart; deleting the key
goes back to the configured ones. A server that fails is tried after the others for a minute.

In `chain` mode a validator needs a validator permit, at least 10,000 total stake and at least 20
alpha of its own.

## Public endpoints

Anyone can read these; the [UI](../ui/README.md) is built on them. Every answer is signed
([Signed reads](#verifiable-log)).

| Endpoint | |
| --- | --- |
| `GET /v1/overview` | the queue of each kind, the work in progress and, for one `kind` (`crawl` unless asked), the window's verdicts, claims and refusals, how long uploads waited for their verdict, verdicts by reason and take-backs |
| `GET /v1/live` | the tasks miners hold now and the uploads being checked, with their size and who has voted |
| `GET /v1/stats/hours` | each hour of one `kind`, the current hour last: tasks handed out, completed and sent back, claims refused by reason, verdicts and credited work, and how many uploads were checked and how long they waited for their verdict; for everyone or one `miner` |
| `GET /v1/stats/series` | tasks, verdicts and credited work of one `kind` per 5, 15 or 60 minutes; for everyone, one `miner` or one `validator` |
| `GET /v1/tasks` | checked tasks with the rows each paid, newest first; filter with `miner`, `validator`, `verdict`, `kind`, `since` |
| `GET /v1/tasks/{task_id}` | one task's state and result, every validator's vote, and for 2 days its per-URL detail (crawl) or sampled texts with their similarity (embed) |
| `GET /v1/votes` | every validator's vote on every checked upload; filter with `validator`, `miner`, `task_id`, `verdict`, `agreed` |
| `GET /v1/miners` | every miner's budget, coverage, share and results in the window, for one `kind` (`crawl` unless asked) |
| `GET /v1/miners/{hotkey}` | a miner's budget of each kind and its history, coverage, each kind's share, results and claims, its take-backs, and lockouts |
| `GET /v1/miners/{hotkey}/verdicts` | signed by that miner: its own finalized uploads, without the ledger delay |
| `GET /v1/validators` | every validator's votes, how often it agreed with the final result, and its standing |
| `GET /v1/validators/{hotkey}` | the same for one validator |
| `GET /v1/events` | the signed log as a feed: tasks issued, completed, refused and returned; filter with `miner`, `task_id`, `outcome`, `request` (a signed request's hash) |
| `GET /v1/health` | queue depths, backlog, the active validators and every validator's standing |
| `GET /v1/room` | how many tasks the queue can take now, for the bot |
| `GET /v1/shares` | every miner's share of the paid work as the API counts it, per pool |
| `GET /v1/rounds`, `GET /v1/rounds/{id}` | round commitments, kept a week after a round closes |
| `GET /v1/rounds/{id}/log` | a round's signed log |
| `GET /v1/key` | the key that signs the log and the notes next to uploads; validators compare it with the key their release pins and never take it from here |
| `GET /v1/storage/probes` | the storage canary's probes since `since` (the last hour), newest first, and the outages they show |

A refused claim is logged once a minute per miner and reason, so an hour's refusals divided by 60 is
how many miners were turned away each minute on average.

Every miner and validator is named by hotkey, with its uid and coldkey beside it (`miner_uid` and
`miner_coldkey`, `validator_uid` and `validator_coldkey`, or `uid` and `coldkey`); both are null for a
hotkey that is not on the metagraph.

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
  validation/open.json  the uploads waiting for validators, with their manifests, rewritten as they change
  log/uploads/          every completed upload with its reported counts, numbered and signed
  log/journal/          the hash-chained log of every change to the accounts, numbered and signed, with the
                        blobs its entries name
  outcomes/             what became of every URL, numbered under seq/, newest in latest.json
  embed/inputs/         the planner's input files, deleted once their vectors are published
desearch-pages bucket
  changes/                every new, changed or removed page with its full record, naming the miner
                          that crawled it and the validator that checked it; numbered under seq/,
                          newest in latest.json; kept ten days
  current/                the latest version of every page, compacted from changes/: 1,024 shards by
                          URL hash, each a sorted base file plus delta files and a manifest naming them
  rounds/                 each round's URL lists, so the URLs of uploads taken back go to the bot again;
                          kept three days
  reports/                every final result, with the checked pages and each validator's report;
                          kept 30 days, the API keeps them all
  vectors/<model>/        vectors, one file per task under log/, numbered under seq/, newest in
                          latest.json; a withdrawal entry drops a task taken back (see the
                          publisher's README)
```

A page's key comes from its URL alone; it is how the change files and `current/` name a page:

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

The Rust tests (`cargo test`) use Redis db 13 at `TASK_API_TEST_REDIS_ADDR` (`127.0.0.1:6379` unless set,
so checkouts testing at once can each run their own). `TASK_API_TEST_CHAIN=test` (or a local node's URL)
also checks an anchor's extrinsic with the runtime's `validate_transaction`; with `TASK_API_TEST_ANCHOR_KEY`
and `TASK_API_TEST_NETUID` on a local chain it is sent and read back. Nothing is ever sent to finney.
