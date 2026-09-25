# Validator Setup

The validator is one process, [`neurons/validators/validator.py`](../neurons/validators/validator.py),
started by `run.sh`. It does two jobs:

1. **Checks crawl tasks.** It reads the list of open uploads from the public uploads bucket,
   downloads each upload from there, re-fetches the picked pages itself, the same pages a chain
   block's hash picks for every validator, and reports pass or fail to the task API with what it
   found for each URL. Every validator checks every upload.
2. **Sets weights.** Each epoch it computes every miner's share from the results of its own checks
   over the last 24 hours and sets weights by those; see [Emission](./emission.md).

`run.sh` checks for a new release every 20 minutes, installs it and restarts the validator.

## Requirements

- Python 3.10 or newer, [PM2](https://pm2.io/docs/runtime/guide/installation/) and `jq`
- A hotkey on netuid 22 (netuid 41 on testnet) with a validator permit and at least 1000 stake; the
  task API only accepts check results from such hotkeys
- A [ScrapingDog](https://www.scrapingdog.com/) API key: the validator fetches each sample itself
  first and uses ScrapingDog only for pages its own address cannot load
- A [Weights & Biases](https://wandb.ai/) login, unless you pass `--wandb.off`

## 1. Install

```bash
git clone https://github.com/Desearch-ai/subnet-22.git
cd subnet-22
python3 -m pip install -r requirements.txt
python3 -m pip install -e .
sudo apt update && sudo apt install -y jq npm && sudo npm install -g pm2
```

## 2. Register a hotkey

```bash
btcli wallet new_coldkey --wallet.name validator
btcli wallet new_hotkey --wallet.name validator --wallet.hotkey default
btcli subnet register --netuid 22 --wallet.name validator --wallet.hotkey default --subtensor.network finney
```

## 3. Configure

```bash
cp neurons/validators/.env.template neurons/validators/.env
```

| Variable | |
| --- | --- |
| `SCRAPINGDOG_API_KEY` | required: a validator that cannot check crawl tasks sets no weights |
| `WANDB_API_KEY` | Weights & Biases login, unless `--wandb.off`; `wandb login` also stores it |
| `EMBED_API_KEY` | for checking [embed tasks](./embedding-tasks.md) once they open: a key for the hosted model service |
| `EMBED_API_URL` | the hosted model service, OpenRouter's `/embeddings` by default |
| `EMBED_PROVIDERS` | which OpenRouter providers run the reference model, `DeepInfra,Nebius` by default |

Nothing else is configurable. How many pages are sampled, how many must match and how many tasks are
checked at once are fixed in code, so every validator checks the same way. Uploads are decoded and
scored only in a memory-capped child process, so an upload built to exhaust memory or time kills
that child, not the validator.

Weights are set only while the checker is healthy. When ScrapingDog refuses three tasks in a row, or
three tasks in a row cannot be scored, the validator logs why and sets no weights until a task is
checked again.

## 4. Run

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
| `--neuron.task_api_url` | the task API, `https://task-api.desearch.ai` by default |
| `--neuron.storage_url` | the public URL of the uploads bucket, `https://r2.desearch.ai` by default |
| `--neuron.disable_set_weights` | check tasks without setting weights |
| `--wandb.off` | do not log to Weights & Biases |
| `--logging.info`, `--logging.debug` | without one, only warnings are printed |

## Upgrading from the search validator

Nothing to do by hand: a validator started with `run.sh` pulls this release, installs it, removes the
old API process and restarts as the single validator process with the same flags. It keeps the
`SCRAPINGDOG_API_KEY` it was started with. Without one it checks nothing and sets no weights, and
logs so, until the key is in `neurons/validators/.env` and the validator restarts.

## Monitor

```bash
pm2 logs desearch_validator_process
curl -s https://task-api.desearch.ai/v1/health
curl -s "https://task-api.desearch.ai/v1/tasks?validator=<hotkey>"
```

`/v1/health` lists the active validators and every validator's standing: how many finalized uploads
it took part in and how many times it disagreed with the majority. A validator that disagrees too
often stops receiving uploads.
