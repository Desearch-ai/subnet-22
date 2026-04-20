# Running a Desearch Miner

A miner runs two required processes, both bound to the same hotkey:

1. **Axon** (`neurons/miners/miner.py`) — Bittensor axon that answers `IsAlive` pings from
   validators. Miners that do not respond to `IsAlive` are marked unreachable and excluded
   from scoring.
2. **Worker API** (`neurons/miners/api.py`) — HTTP service that receives signed search
   requests (AI / Twitter / Web) pushed by validators and returns results. This is the
   scoring path.

## Prerequisites

- Python ≥ 3.10
- [PM2](https://pm2.io/docs/runtime/guide/installation/) for process supervision
- A registered hotkey on subnet 22 (mainnet) or 41 (testnet)
- Environment variables configured — see [env_variables.md](./env_variables.md)

## Install

```sh
git clone https://github.com/Desearch-ai/subnet-22.git
cd subnet-22
python -m pip install -r requirements.txt
python -m pip install -e .
```

## Configure the worker manifest

Copy the template and edit for your deployment:

```sh
cp neurons/miners/workers.template.json neurons/miners/workers.json
```

Example `neurons/miners/workers.json`:

```json
{
  "worker_url": "http://127.0.0.1:8000",
  "concurrency": {
    "web_search": 20,
    "x_search": 15,
    "ai_search": 10
  }
}
```

- `worker_url` — single endpoint that validators call. Place your load balancer or multiple
  instances behind this URL.
- `concurrency` — **per search type, per validator** ceiling. With 12 active validators,
  `web_search: 20` means up to `20 × 12 = 240` concurrent web search requests in the worst
  case. Infrastructure sizing is your responsibility.

Updates to `workers.json` propagate via `IsAlive` and take effect at the next UTC hour
boundary without restart.

## Configure env vars

```sh
cp neurons/miners/.env.template neurons/miners/.env
# edit neurons/miners/.env
```

See [env_variables.md](./env_variables.md) for the full list.

## Run with PM2

Both processes are required. Run them on the same host with the same wallet + hotkey.

### Axon

```sh
pm2 start neurons/miners/miner.py \
  --interpreter /usr/bin/python3 \
  --name desearch_miner_axon \
  -- \
  --wallet.name miner \
  --wallet.hotkey default \
  --subtensor.network finney \
  --netuid 22 \
  --axon.port 14000
```

### Worker API

```sh
pm2 start neurons/miners/api.py \
  --interpreter /usr/bin/python3 \
  --name desearch_miner_worker \
  -- \
  --host 0.0.0.0 \
  --port 8000 \
  --wallet.name miner \
  --wallet.hotkey default \
  --subtensor.network finney \
  --netuid 22
```

The worker API enforces signature auth on every request. Validators are accepted only if
they are registered, hold a validator permit, and meet the minimum stake thresholds
(`MIN_ALPHA_STAKE` and `MIN_TOTAL_STAKE` in `desearch/__init__.py`).

### Key flags

- `--wallet.name` / `--wallet.hotkey` — registered wallet + hotkey
- `--netuid` — `22` mainnet, `41` testnet
- `--subtensor.network` — `finney`, `test`, or custom endpoint
- `--axon.port` — public port for the axon (miner only)
- `--host` / `--port` — worker API bind address and port
- `--miner.config_path` — path to `workers.json` (default `./neurons/miners/workers.json`)
- `--logging.debug` / `--logging.trace` — increase log verbosity

## Monitor

```sh
pm2 status
pm2 logs desearch_miner_axon
pm2 logs desearch_miner_worker
```
