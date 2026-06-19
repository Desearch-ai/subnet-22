<div align="center">

<img src="./docs/assets/desearch-logo.png" alt="Desearch" width="480" />

# Subnet 22 (SN22) on Bittensor

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

</div>

SN22 is the Bittensor subnet behind Desearch's decentralized real-time intelligence layer. It coordinates miners and validators that serve live web, X/Twitter, and multi-source search data for Desearch API and console products.

This repository contains the subnet runtime: miner axons, validator scoring, the validator FastAPI service, operator runbooks, and shared protocol models.

## Architecture

| Component | Role | Main files |
| --- | --- | --- |
| Miner | Runs a Bittensor axon, declares search capacity, answers health checks, and serves AI, X/Twitter, and web search synapses. | `neurons/miners/miner.py`, `neurons/miners/config.py`, `neurons/miners/manifest.template.json` |
| Validator | Sends synthetic and organic queries to miners, verifies results with independent providers, stores scoring windows, and writes weights on-chain. | `neurons/validators/validator_service.py`, `neurons/validators/scoring/`, `neurons/validators/reward/` |
| Validator API | Runs FastAPI next to the validator so trusted Desearch services can request organic search and inspect public miner state. Protected routes require the `access-key` header. | `neurons/validators/api.py`, `neurons/validators/dependencies.py`, `run.sh` |
| Shared package | Defines protocol models, synapses, tool helpers, dataset utilities, and common configuration. | `desearch/protocol.py`, `desearch/miner_config.py`, `desearch/tools/` |

## Setup path

### 1. Install the subnet package

```bash
git clone https://github.com/Desearch-ai/subnet-22.git
cd subnet-22
python3 -m pip install -r requirements.txt
python3 -m pip install -e .
```

The root project is Python (`requirements.txt`, `setup.py`). `utility-api/` is a separate FastAPI service with its own `pyproject.toml`.

### 2. Choose an operator role

- **Miner operators** register a hotkey on netuid 22, configure `neurons/miners/.env`, create `neurons/miners/manifest.json`, and run the miner axon under PM2. See [Running a Miner](./docs/running_a_miner.md).
- **Validator operators** register a validator hotkey on netuid 22, configure validator credentials, run `run.sh` to manage the validator service plus API, and monitor weights/scoring. See [Running a Validator](./docs/running_a_validator.md).
- **Desearch service integrators** should use the external Desearch API/console. The validator API documented here is a subnet-facing service protected by `EXPECTED_ACCESS_KEY`, not the public billing/auth layer.

### 3. Configure environment variables

See [Environment Variables](./docs/env_variables.md) for shared, miner-only, and validator-only settings.

Common requirements:

- Shared: `OPENAI_API_KEY`, `APIFY_API_KEY`, `SCRAPINGDOG_API_KEY`
- Miner-only: optional `TWITTER_BEARER_TOKEN`, wallet/netuid/axon settings
- Validator-only: `EXPECTED_ACCESS_KEY`, `WANDB_API_KEY`, API/service ports

### 4. Validate the checkout

```bash
# Fast syntax/import check
python3 -m compileall desearch neurons tests scripts

# Project test suite
pytest

# Confirm the current validator link endpoints are present in source
grep -n '"/search/links/web"\|"/search/links/twitter"\|"/search/links"' neurons/validators/api.py

# Runtime checks after PM2 processes are running
pm2 status
pm2 logs desearch_miner
pm2 logs desearch_validator_process
pm2 logs desearch_api_process
```

## Current validator API surface

The validator API is implemented in `neurons/validators/api.py` and documented in [API Reference](./docs/api.md). Key search-link routes are:

- `POST /search/links/web`
- `POST /search/links/twitter`
- `POST /search/links`

Protected routes require the `access-key` header matching `EXPECTED_ACCESS_KEY`. Public miner status routes under `/public/miners` are intentionally unauthenticated.

## Documentation index

- [API Reference](./docs/api.md) — validator API routes, auth, request shapes, and examples.
- [Environment Variables](./docs/env_variables.md) — shared, miner-only, and validator-only variables.
- [Running a Miner](./docs/running_a_miner.md) — miner install, manifest, PM2 run commands, and monitoring.
- [Running a Validator](./docs/running_a_validator.md) — validator service/API/autoupdate processes and operational flags.
- [Mainnet Operations](./docs/running_on_mainnet.md) — running miner or validator hotkeys on Bittensor mainnet netuid 22.
- [Testnet Operations](./docs/running_on_testnet.md) — testnet wallet/subnet workflow.

## Support

- Website: [desearch.ai](https://desearch.ai)
- Console/API product: [console.desearch.ai](https://console.desearch.ai)
- Desearch Discord: [Join the community](https://discord.com/invite/eb6DTZNMF5)
