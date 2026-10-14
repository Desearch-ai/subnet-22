# 🛡️ Validator

- [🔍 What a validator does](#-what-a-validator-does)
- [🔧 Setup](#-setup)

## 🔍 What a validator does

A validator checks miners' work under rules fixed in code, so every validator checks the same way, and
sets weights from what its own checks found. It is one process,
[`neurons/validators/validator.py`](../neurons/validators/validator.py), started by `run.sh`, with three
jobs:

1. **Checks crawl tasks.**
   - A chain block's hash draws a share of every miner's uploads for a check, and the task API lists
     those in the public uploads bucket.
   - The validator reads each listed upload and re-fetches the pages the same hash picks; the picks are
     the same for every validator.
   - It reports pass or fail to the task API, with what it found per URL. Every validator checks every
     listed upload.
2. **Checks embed tasks.**
   - Every embed upload is checked.
   - The validator samples 48 of its texts, gets their vectors from the same model and compares them
     with the miner's. Each must reach a cosine similarity of 0.999.
   - The vectors come from the validator's own model server. A validator without one falls back to
     Desearch's model server through the task API, which embeds each sampled text once.
3. **Sets weights.**
   - Each epoch, 50% of the emission goes to burn, 20% to crawl miners and 30% to embed miners; a part
     no miner earned is burned.
   - Crawl miners are paid from the log of every completed upload, at the rate the validator's own
     checks of each miner paid, less what its failed checks take back.
   - Embed miners are paid by the characters their passed tasks embedded over the last 24 hours, as
     the validator's own checks found.
   - See [Emission](./emission.md).

**Healthy checking:** weights are set only while the validator is checking. It sets none, and logs why,
until a task is checked again, when any of these happens:
- ScrapingDog refuses three tasks in a row;
- three tasks in a row cannot be scored;
- the task API refuses the validator's reports.

**Updates:** `run.sh` checks for a new release every 20 minutes, installs it and restarts the
validator.

## 🔧 Setup

### Requirements

- Python 3.10 or newer, [PM2](https://pm2.io/docs/runtime/guide/installation/) and `jq`
- A hotkey on netuid 22 (netuid 41 on testnet) with a validator permit, at least 10,000 total stake
  and at least 20 alpha; the task API only accepts check results from such hotkeys
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
sudo apt update && sudo apt install -y jq npm && sudo npm install -g pm2
```

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
| `EMBED_API_URL` | optional: your own model server's `/v1/embeddings` for checking [embed tasks](./embedding-tasks.md); without it the vectors come from the task API |
| `EMBED_API_KEY` | if your server requires one |

Nothing else is configurable.

**Your own model server** runs the same `vllm serve` command miners use, from
[Embedding tasks](./embedding-tasks.md), on an NVIDIA GPU with 16 GB of memory or more. The validator
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
| `--neuron.storage_url` | the public URL of the uploads bucket, `https://r2.desearch.ai` by default |
| `--neuron.disable_set_weights` | check tasks without setting weights |
| `--wandb.off` | do not log to Weights & Biases |
| `--logging.info`, `--logging.debug` | without one, only warnings are printed |

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
it took part in, and how many times it disagreed with the majority. A validator that disagrees too
often stops receiving uploads.
