# Running a Desearch Validator

A validator runs three PM2 processes (managed by `run.sh`):

1. `desearch_validator_process` — core neuron (`neurons/validators/validator_service.py`).
   Syncs the metagraph, generates synthetic queries, dispatches to miners via dendrite,
   scores responses, and writes weights on-chain.
2. `desearch_api_process` — public FastAPI (`neurons/validators/api.py`) that serves organic
   search requests to paying consumers.
3. `desearch_autoupdate` — `run.sh` itself, which pulls new releases every 20 minutes and
   restarts the other two processes.

## Prerequisites

- Python ≥ 3.11 (recommended: conda env)
- [Redis](https://redis.io/docs/latest/operate/oss_and_stack/install/install-redis/)
- [PM2](https://pm2.io/docs/runtime/guide/installation/)
- `jq`, `npm` for the autoupdate loop
- A registered validator hotkey on netuid 22 (mainnet) or 41 (testnet)

### SQLite

Miner scoring state is persisted to `.state/miner_state.db` (auto-created on startup,
3-day rolling retention). No manual SQLite setup is required: `aiosqlite` is installed
via `requirements.txt` and uses Python's bundled `sqlite3` stdlib module. The `.state/`
directory is under the repo root and gitignored.

## Install

Create a conda env, clone the repo, install deps:

```sh
conda create -n val python=3.11 -y
conda activate val

git clone https://github.com/Desearch-ai/subnet-22.git
cd subnet-22
python -m pip install -r requirements.txt
python -m pip install -e .
```

System packages (Ubuntu/Debian):

```sh
sudo apt update && sudo apt install -y jq npm
sudo npm install -g pm2 && pm2 update
```

macOS:

```sh
brew update && brew install jq npm
sudo npm install -g pm2 && pm2 update
```

## Configure env vars

See [env_variables.md](./env_variables.md). At minimum export:

```sh
export OPENAI_API_KEY="…"
export APIFY_API_KEY="…"
export SCRAPINGDOG_API_KEY="…"
export WANDB_API_KEY="…"
export EXPECTED_ACCESS_KEY="$(python scripts/generate_access_key.py)"
```

`EXPECTED_ACCESS_KEY` gates the validator's organic-search FastAPI. The Desearch
backend calls this API on behalf of API consumers using the key as the `access-key`
header — end users never see it. The generator script produces a key that satisfies
the validator's length and character-class requirements (≥16 chars, uppercase +
lowercase + digit + special char). Share it with the Desearch team.

Run `wandb login` once before starting the validator.

## Run

Recommended entry point — autoupdate loop:

```sh
pm2 start run.sh --name desearch_autoupdate -- \
  --wallet.name <wallet-name> \
  --wallet.hotkey <hotkey-name> \
  --netuid 22 \
  --subtensor.network finney
```

Tune the API:

```sh
pm2 start run.sh --name desearch_autoupdate -- \
  --workers 4 --port 8005 \
  --wallet.name <wallet-name> \
  --wallet.hotkey <hotkey-name> \
  --netuid 22 \
  --subtensor.network finney
```

Manual run (skip autoupdate):

```sh
pm2 start neurons/validators/validator_service.py \
  --interpreter /usr/bin/python3 \
  --name desearch_validator_process \
  -- \
  --wallet.name <wallet-name> \
  --wallet.hotkey <hotkey-name> \
  --netuid 22 \
  --subtensor.network finney

pm2 start uvicorn \
  --interpreter /usr/bin/python3 \
  --name desearch_api_process \
  -- \
  neurons.validators.api:app \
  --host 0.0.0.0 --port 8005 --workers 4
```

### Key flags

- `--wallet.name` / `--wallet.hotkey` — validator wallet + hotkey
- `--netuid` — `22` mainnet, `41` testnet
- `--subtensor.network` — `finney`, `test`, or custom endpoint
- `--neuron.device` — `cuda` or `cpu`
- `--neuron.scoring_model` — LLM used for scoring. Default `openai/gpt-4.1-nano`. Also
  accepts Qwen, Mistral, DeepSeek variants.
- `--neuron.disable_log_rewards` — suppress per-reward wandb logs (default `False`)

## Monitor

```sh
pm2 status
pm2 logs desearch_validator_process
pm2 logs desearch_api_process
pm2 logs desearch_autoupdate
```

Metrics are streamed to W&B at
https://wandb.ai/smart-scrape/smart-scrape-1.0 by default.

> Allocate at least 50 GB of free disk for W&B logs.
