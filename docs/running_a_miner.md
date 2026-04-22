# Running a Desearch Miner

A miner runs a single process — a Bittensor axon (`neurons/miners/miner.py`) that answers
`IsAlive` pings and all search synapses (AI / Twitter / Web). Validators call the axon
directly via dendrite.

## Prerequisites

- Python ≥ 3.10
- [PM2](https://pm2.io/docs/runtime/guide/installation/) for process supervision
- A registered hotkey on subnet 22 (mainnet) or 41 (testnet)

## Install

```sh
git clone https://github.com/Desearch-ai/subnet-22.git
cd subnet-22
python -m pip install -r requirements.txt
python -m pip install -e .
```

## Configure the miner manifest

Copy the template and edit for your deployment:

```sh
cp neurons/miners/manifest.template.json neurons/miners/manifest.json
```

Example `neurons/miners/manifest.json`:

```json
{
  "concurrency": {
    "web_search": 20,
    "x_search": 15,
    "ai_search": 10
  }
}
```

- `concurrency` — **per search type, per validator** ceiling. With 12 active validators,
  `web_search: 20` means up to `20 × 12 = 240` concurrent web search requests in the worst
  case. Infrastructure sizing is your responsibility (scale the axon host, or front it
  with a load balancer routing to multiple backends).

Updates to `manifest.json` propagate via `IsAlive` and take effect at the next UTC hour
boundary without restart.

## Configure env vars

The miner loads `neurons/miners/.env` automatically on startup. Copy the template and
fill it in:

```sh
cp neurons/miners/.env.template neurons/miners/.env
# edit neurons/miners/.env
```

See [env_variables.md](./env_variables.md) for the full list.

## Run with PM2

Two equivalent ways — pick whichever fits your workflow.

### Option A: `.env` (recommended)

All runtime config comes from `neurons/miners/.env`:

```sh
pm2 start neurons/miners/miner.py \
  --interpreter /usr/bin/python3 \
  --name desearch_miner
```

### Option B: CLI-flag-driven

Pass runtime config on the command line (overrides anything in `.env`):

```sh
pm2 start neurons/miners/miner.py \
  --interpreter /usr/bin/python3 \
  --name desearch_miner \
  -- \
  --wallet.name miner \
  --wallet.hotkey default \
  --subtensor.network finney \
  --netuid 22 \
  --axon.port 14000
```

## Stake

Incoming synapses are stake-gated at the axon: the sender must be registered, hold a
validator permit, and meet the minimum stake thresholds (`MIN_ALPHA_STAKE` and
`MIN_TOTAL_STAKE` in `desearch/__init__.py`).

### Key flags

- `--wallet.name` / `--wallet.hotkey` — registered wallet + hotkey
- `--netuid` — `22` mainnet, `41` testnet
- `--subtensor.network` — `finney`, `test`, or custom endpoint
- `--axon.port` — public port for the axon
- `--miner.config_path` — path to `manifest.json` (default `./neurons/miners/manifest.json`)
- `--logging.debug` / `--logging.trace` — increase log verbosity

## Monitor

```sh
pm2 status
pm2 logs desearch_miner
```
