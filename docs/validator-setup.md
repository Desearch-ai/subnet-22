# Validator Setup

The validator is one process, [`neurons/validators/validator.py`](../neurons/validators/validator.py),
started by `run.sh`. It does two jobs:

1. **Checks crawl tasks.** It leases completed tasks from the task API, downloads each upload,
   re-fetches a sample of the pages itself and returns a verdict with what it found for each URL.
2. **Sets weights.** Each epoch it reads each pool's shares from the task API and sets weights: each
   pool paid out by share, the rest to the subnet's burn hotkey. Crawl is the only pool today, with
   half the emission.

`run.sh` checks for a new release every 20 minutes, installs it and restarts the validator.

## Requirements

- Python 3.10 or newer, [PM2](https://pm2.io/docs/runtime/guide/installation/) and `jq`
- A hotkey on netuid 22 (netuid 41 on testnet) with a validator permit and at least 1000 stake; the
  task API only accepts verdicts from such hotkeys
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
| `SCRAPINGDOG_API_KEY` | required to check crawl tasks; without it the validator only sets weights |
| `WANDB_API_KEY` | Weights & Biases login, unless `--wandb.off`; `wandb login` also stores it |
| `EMBED_API_KEY` | required to check embed tasks: a key for `EMBED_API_URL` |
| `EMBED_API_URL` | where the reference vectors come from, OpenRouter's `/embeddings` by default |
| `EMBED_PROVIDERS` | OpenRouter providers to pin, `DeepInfra,Nebius` by default |

Nothing else is configurable. How many pages are sampled, how many must match and how many tasks are
checked at once are fixed in code, so every validator checks the same way. Uploads are decoded and
scored only in a memory-capped child process, so an upload built to exhaust memory or time kills
that child, not the validator.

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
| `--wallet.name`, `--wallet.hotkey` | the validator's wallet; it also signs verdicts |
| `--netuid` | `22` on mainnet, `41` on testnet |
| `--subtensor.network` | `finney`, `test`, or a custom endpoint |
| `--neuron.task_api_url` | the task API, `https://task-api.desearch.ai` by default |
| `--neuron.disable_set_weights` | check tasks without setting weights |
| `--wandb.off` | do not log to Weights & Biases |
| `--logging.info`, `--logging.debug` | without one, only warnings are printed |

## Upgrading from the search validator

Nothing to do by hand: a validator started with `run.sh` pulls this release, installs it, removes the
old API process and restarts as the single validator process with the same flags. It keeps the
`SCRAPINGDOG_API_KEY` it was started with. Without one it keeps setting weights and logs that it is
not checking crawl tasks until the key is in `neurons/validators/.env` and the validator restarts.

## Monitor

```bash
pm2 logs desearch_validator_process
curl -s https://task-api.desearch.ai/v1/health
curl -s "https://task-api.desearch.ai/v1/tasks?validator=<hotkey>"
```

`/v1/health` lists every validator's audit record: a share of verdicts is checked by a second
validator, and a validator that loses too many of those checks can no longer lease tasks.
