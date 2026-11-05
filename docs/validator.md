# 🛡️ Validator

- [🔍 What a validator does](#-what-a-validator-does)
- [🔐 Which task API it trusts](#-which-task-api-it-trusts)
- [🔧 Setup](#-setup)

## 🔍 What a validator does

A validator checks miners' work under rules fixed in code, so every validator checks the same way, and
sets weights from what its own checks found. It is one process,
[`neurons/validators/validator.py`](../neurons/validators/validator.py), started by `run.sh`, with four
jobs:

1. **Checks crawl tasks.**
   - A chain block's hash draws a share of every miner's uploads for a check, and the task API lists
     those in the public uploads bucket.
   - The validator reads each listed upload and re-fetches the pages the same hash picks; the picks are
     the same for every validator.
   - It reports pass or fail to the task API, with what it found per URL. Every validator checks every
     listed upload.
2. **Checks embed tasks.**
   - The same way, a block hash draws a share of every miner's embed uploads, and the task API lists
     those too.
   - For each listed upload, the validator samples 48 of its texts and reads only their vectors from
     the upload. It gets vectors for the same texts from the same model and compares. Each must reach
     a cosine similarity of 0.999.
   - The vectors come from the validator's own model server. A validator without one falls back to
     Desearch's model server through the task API, which embeds each sampled text once.
3. **Sets weights.**
   - Each epoch, 30% of the emission goes to burn, 20% to crawl miners and 50% to embed miners; a part
     no miner earned is burned.
   - Crawl miners are paid from the log of every completed upload, at the rate the validator's own
     checks of each miner paid, less what its failed checks take back.
   - Embed miners are paid the same way, counted in the characters of the texts they embedded.
   - Coverage is relative to the round, as the task API judges it: a task must return 85% × the
     share of the round's pages validators could load, counting only URLs and rows outside the
     round's **neutral hosts** (hosts most checked uploads of the round could not load). Those hosts
     pay nobody and cost nobody. Once a round closes, the gate and neutral hosts are the ones the
     task API settled in its signed log (the round's `close` entry, read by the log follower).
     While the round is open, the validator uses its own checks of it: their samples, once there
     are 30, and the hosts most of them could not load. Your vote is cast on that view. Your
     weights judge coverage again each time they are set, so a check made early in a round is paid
     on the round's final gate.
   - An upload that storage served in another version than the one its completion pinned
     (`upload_changed`) is still reported, so the task API keeps the evidence against storage and
     never publishes the upload. Your weights neither pay nor charge the miner for it, and do not pay
     it as an unchecked upload either.
   - See [Emission](./emission.md).

4. **Checks the task API itself.**
   - The task API signs with a key this release pins (`desearch/trust.py`); the validator never takes the
     key from the API's own `/v1/key`, only compares it and logs a mismatch.
   - It follows the API's hash-chained log in the uploads bucket: every file must be signed by the pinned
     key and follow on from the last, and every block hash an entry decided on must be the one your own node
     has.
   - It reads every anchor the API's anchor hotkey sent from the blocks on your own node: the anchored entry
     must never go back and each anchored head must be the log's own hash, so a rolled-back or forked log
     shows even if it happened before your validator started. A log file taken down within the bucket's day,
     whose entries an anchor covers, is fraud too. The anchor hotkey is the one this release pins
     (`TASK_API_ANCHOR_HOTKEY`, or `--neuron.task_api_anchor`) and is never the log's signer; one naming the
     signer is ignored. Until one is pinned, anchors are not checked: the validator logs that at every
     poll, publishes it beside its proofs, and sets weights from its own checks only, since an unanchored
     log can be rewritten without a trace.
   - It checks every metagraph the API logged (coldkeys, permits, stakes, incentives) against your node at
     the block it names.
   - It replays the log with the API's own rules (`task-api-replay`, which `run.sh` builds) and reports
     every budget, verdict, take-back or share the rules do not give, and a round committed again although
     its commitment was anchored in time.
   - Whatever does not hold is logged as an error and written as a fraud proof (the signed file, entries and
     chain reads that show it) under `~/.bittensor/miners/<wallet>/<hotkey>/netuid22/validator/journal/fraud/`,
     published by your relay at `/v1/journal/proofs` and in your Weights & Biases run.
   - From a fraud until the log holds together again (the replay, started again at a snapshot past it,
     holds through an anchor on chain), and whenever the replay cannot run or keep up, the validator sets
     weights from its own checks only: the API's upload log, budgets, verdicts and shares reach no weight,
     and uploads logged while the log showed fraud are never paid.
     See [Verifiable log](../task-api/README.md#verifiable-log).

**Healthy checking:** weights are set only while the validator is checking. It sets none, and logs why,
until a task is checked again, when any of these happens:
- ScrapingDog refuses three tasks in a row;
- three tasks in a row cannot be scored;
- the task API refuses the validator's reports.

**Updates:** `run.sh` checks for a new release every 20 minutes, installs it and restarts the
validator.

## 🔐 Which task API it trusts

The task API runs in a TEE, and the validator follows only a key it has checked itself
([how it works](./tee.md)):

- It reads the API's **custody records** from the uploads bucket (`log/custody/`): the chain of keys the
  log has been signed with. It checks each new record's signature and link, that validators' stake had
  approved its version at its block, and the new key's attestation: Intel's signature on the TDX quote,
  the pinned production OS, the compose (images pinned by digest, only credentials set from outside, every
  protection on) and the key bound into the quote.
- It takes each upload log file and manifest only from the key whose turn it was, and stops following
  the API when its version loses validators' endorsement or when two valid records claim one place in
  the chain. Accepted records are kept in `custody.json` in the validator's folder.
- Until the API publishes custody records, it follows only the signer this release pins
  (`desearch/trust.py`, or `--neuron.legacy_signer`); with none it follows nothing and logs an error. It
  never takes the key the task API names at `/v1/key`, which whoever answers the domain could choose.

**Approving a version.** A new version takes effect only once more than half the stake of validators
independent of the subnet owner, from at least three coldkeys, has listed its compose hash in their
commitment for 7 days (50,400 blocks). The owner's own hotkeys, validators inheriting stake from them, and
the stake the owner's coldkey delegated to any validator never count ([the rule](./tee.md#approving-a-version)). Check the version first ([verify](./verify-task-api.md)), then commit it with the validator's
hotkey; the tool prints the commitment and sends nothing without `--commit`:

```bash
python3 tools/approve_version.py --version <running compose hash> --next <new compose hash>
python3 tools/approve_version.py --version <running> --next <new> --commit --network finney \
  --wallet.name validator --wallet.hotkey default
```

- Keep the running version listed (`--version`) while its successor waits: a commitment that drops
  it ends the running version's endorsement at once.
- To start the chain, or to replace a CVM lost for good, appoint one CVM of an approved version with
  `--version <hash> --appoint <its signer>`, after checking its attestation.
- A commitment replaces the hotkey's last one on the subnet.

## 🔧 Setup

### Requirements

- Python 3.10 or newer, [PM2](https://pm2.io/docs/runtime/guide/installation/) and `jq`
- [Rust](https://rustup.rs) and a C compiler (`build-essential`): `run.sh` builds the task API's replay
  tool from the release, and a validator that sets weights does not start without it
- A hotkey on netuid 22 (netuid 41 on testnet) with a validator permit, at least 10,000 total stake
  and at least 20 alpha; the task API only accepts check results from such hotkeys. These are
  mainnet's minimums; a task API on testnet may set lower ones (`TASK_API_VALIDATOR_MIN_STAKE`,
  `TASK_API_VALIDATOR_MIN_ALPHA`), and its `/v1/config` shows them
- A [ScrapingDog](https://www.scrapingdog.com/) API key: the validator fetches each sample itself
  first and uses ScrapingDog only for pages its own address cannot load
- A [Weights & Biases](https://wandb.ai/) login, unless you pass `--wandb.off`
- For embed checks, your own model server on a GPU (see Configure); without one, the validator falls
  back to Desearch's model server

### 1. Install

```bash
git clone https://github.com/Desearch-ai/subnet-22.git
cd subnet-22
python3 -m pip install -r requirements.txt
python3 -m pip install -e .
sudo apt update && sudo apt install -y jq npm build-essential && sudo npm install -g pm2
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
```

`run.sh` builds `task-api/target/release/task-api-replay` before it starts the validator, and again after
every update, so the replay always has the release's own rules.

### 2. Register a hotkey

```bash
btcli wallet new-coldkey --wallet validator
btcli wallet new-hotkey --wallet validator --wallet-hotkey default
btcli subnets register --netuid 22 --wallet validator --wallet-hotkey default --network finney
```

### 3. Configure

```bash
cp neurons/validators/.env.template neurons/validators/.env
```

| Variable | |
| --- | --- |
| `SCRAPINGDOG_API_KEY` | required: a validator that cannot check crawl tasks sets no weights |
| `SCRAPINGDOG_DAILY_CREDITS` | ScrapingDog credits to use per day, spread evenly over the hours; 100,000 by default |
| `WANDB_API_KEY` | Weights & Biases login, unless `--wandb.off`; `wandb login` also stores it |
| `EMBED_API_URL` | optional: your own model server's `/v1/embeddings` for checking [embed tasks](./embedding-tasks.md); without it your validator does not check embed tasks, since reference vectors come only from validators' own models |
| `EMBED_API_KEY` | if your server requires one |

Nothing else is configurable.

**Your own model server** runs the same `vllm serve` command miners use, from
[Miner → Embedding](./miner.md#5-embedding), on an NVIDIA GPU with 16 GB of memory or more. The validator
checks which models the server answers to, and waits for it when it is not up yet.

**Sandboxed checks:** uploads are checked in a separate, memory-capped process.
- On Linux that process also has no environment, no network and read-only access to Python's files
  only, so it cannot reach your wallet, keys or the internet.
- This needs unprivileged user namespaces or Landlock, which most distributions provide. Docker's
  default seccomp profile blocks both; the validator logs a warning when running without them.

**ScrapingDog credits:** keep them topped up. With an invalid key or no credits left, the validator
stops checking and setting weights until the account works again. It warns a day before credits run
out.

### 4. Run

```bash
wandb login
pm2 start run.sh --name desearch_autoupdate -- \
  --wallet.name validator \
  --wallet.hotkey default \
  --netuid 22 \
  --subtensor.network finney \
  --logging.info
```

| Flag | |
| --- | --- |
| `--wallet.name`, `--wallet.hotkey` | the validator's wallet; it also signs the check results |
| `--netuid` | `22` on mainnet, `41` on testnet |
| `--subtensor.network` | `finney`, `test`, or a custom endpoint |
| `--neuron.task_api_url` | the task API, `https://api-22.desearch.ai` by default |
| `--neuron.storage_url` | the public URL of the uploads bucket, `https://subnet-22.s3.us-east-va.io.cloud.ovh.us` by default |
| `--neuron.disable_set_weights` | check tasks without setting weights |
| `--neuron.task_api_signer` | the key the task API signs with; pinned in each release, change it only with a release |
| `--neuron.task_api_anchor` | the hotkey whose commitment anchors the task API's log; the signer by default |
| `--neuron.replay_bin` | the replay tool, `task-api/target/release/task-api-replay` by default (or `TASK_API_REPLAY_BIN`), as `run.sh` builds it (`cargo build --release --locked --bin task-api-replay` in `task-api/`); it must be this release's version |
| `--neuron.no_replay` | run without the replay: logged at start and at every weight setting, and the API's inputs then never reach the weights, which come from your own checks only |
| `--neuron.archive_network` | a chain endpoint with old state, for approvals at old custody records and for metagraphs the API logged longer ago than your node keeps state; the subtensor's archive fallback when empty |
| `--neuron.tee_allow_any_os`, `--neuron.tee_simulated` | testnets only: accept any OS image, or the dstack simulator's quotes; ignored on finney |
| `--wandb.off` | do not log to Weights & Biases |
| `--logging.info`, `--logging.debug` | without one, only warnings are printed |
| `--relay.port` | serve a relay for miners' requests on this port, signing a witness of each; 0 (the default) serves none |
| `--relay.publish_delay_blocks`, `--relay.check_after_blocks` | blocks before a witness is listed (5), and after which a witnessed request whose outcome the task API's anchored log lacks or contradicts is a censorship proof (50) |
| `--relay.late_blocks` | blocks after forwarding within which the log must have judged a witnessed request (10) |
| `--relay.signer` | the key the task API's log must be signed with; the custody chain's signer (or the pinned one) when empty |
| `--witness.storage_off` | do not look at, and sign what storage holds for, every completed upload |
| `--witness.probe_every_s` | seconds between this validator's own probes of storage through a link the task API hands validators (60); failures, and looks at uploads storage failed, are reported to the API, and enough validators' reports forgive the claims alive then; 0 probes nothing |

**Upload log.** Each upload log file names the hash of the one before; the validator checks the links and keeps a
signed fraud proof when a file it read is forked or one younger than a day is withheld, instead of reading past it.

**Relays and witnesses.** A validator serving a relay gives miners a path to the task API outside the
operator's network, signs a witness of every request it forwards and publishes censorship proofs for any the
API's log then lacks. Every validator also looks at every completed upload in storage and signs what it saw.
Open the relay's port to miners and tell them its URL. See [witness relays](./witness-relays.md).

### Upgrading from the search validator

Nothing to do by hand.
- A validator started with `run.sh` pulls this release, installs it, removes the old API process and
  restarts as the single validator process with the same flags.
- It keeps the `SCRAPINGDOG_API_KEY` it was started with.
- Without a key it checks nothing and sets no weights, and logs so, until the key is in
  `neurons/validators/.env` and the validator restarts.

### 📈 Monitor

```bash
pm2 logs desearch_validator_process
curl -s https://api-22.desearch.ai/v1/health
curl -s "https://api-22.desearch.ai/v1/tasks?validator=<hotkey>"
```

`/v1/health` lists the active validators and every validator's standing: how many finalized uploads
it took part in, how many times it disagreed with the majority, and its wrong reports. Disagreeing
costs nothing: your live-page judgment goes into your own weights. Only reports your tasks could not
have produced (a sampled URL outside the task, rows that do not add up), and reports whose row counts
the upload's own file contradicts against the miner (more duplicates or foreign rows, fewer rows or
more error rows than the file has: the task API counts them itself when they could charge anyone),
count as wrong; five within a day and the API refuses your reports for the rest of it. Reports carry
the facts of each file (rows, duplicates, rows per host, rows not re-extracted from their HTML); the
task API charges miners on those facts alone, only when validators of two different coldkeys report the
same facts, and only at the upload's validation deadline. Facts that differ between two validators
charge nobody. See [Rules](./rules.md).
