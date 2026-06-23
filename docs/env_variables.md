# Environment Variables

Reference for environment variables consumed by SN22 miners and validators. Do not commit real secrets. Export validator settings in the operator environment; miners normally load `neurons/miners/.env` copied from `neurons/miners/.env.template`.

## Obtaining credentials

- **OpenAI** — https://platform.openai.com/ (LLM scoring, summaries, and query handling)
- **Apify** — https://apify.com/ (X/Twitter scraping and validator verification)
- **ScrapingDog** — https://www.scrapingdog.com/ (miner web search and validator web verification)
- **Twitter API** — https://developer.twitter.com/en/portal/dashboard (optional miner direct tweet access)
- **Weights & Biases** — https://wandb.ai/ (validator metrics)

## Shared variables

These can be required by both miner and validator processes, depending on the role being run.

| Variable | Required | Used by | Purpose |
| --- | --- | --- | --- |
| `OPENAI_API_KEY` | yes | miner, validator | Miner summaries/query handling; validator scoring and summary generation. |
| `APIFY_API_KEY` | yes | miner, validator | X/Twitter scraping for miners and validator-side verification. |
| `SCRAPINGDOG_API_KEY` | yes | miner, validator | Miner web search; validator web content verification. |
| `REDIS_HOST` | no | validator utilities | Redis host for helper clients; defaults to `localhost`. |
| `REDIS_PORT` | no | validator utilities | Redis port for helper clients; defaults to `6379`. |
| `CHUTES_API_TOKEN` | no | validator utilities | Optional Chutes LLM fallback for code paths that call `call_chutes`. |

## Validator-only variables

| Variable | Required | Purpose |
| --- | --- | --- |
| `EXPECTED_ACCESS_KEY` | yes for API | Gates protected validator API routes in `neurons/validators/api.py`. Must be at least 16 characters and include uppercase, lowercase, digit, and special character. Generate with `python3 scripts/generate_access_key.py`. |
| `WANDB_API_KEY` | yes | Metrics dashboard login. Run `wandb login` once after installing. |
| `PORT` | no | Validator API port; default `8005`. |
| `VALIDATOR_SERVICE_PORT` | no | IPC port between the API and validator service; default `8006`. |
| `MINER_DB_PATH` | no | Validator miner scoring SQLite path; default `.state/miner_state.db` under the repo root. |

### Validator export example

```bash
export OPENAI_API_KEY="***"
export APIFY_API_KEY="***"
export SCRAPINGDOG_API_KEY="***"
export WANDB_API_KEY="***"
export EXPECTED_ACCESS_KEY="$(python3 scripts/generate_access_key.py)"
```

## Miner-only variables

Miners load `neurons/miners/.env` automatically on startup. CLI args passed to `pm2 start ... -- ...` take precedence over `.env` values.

| Variable | Required | Purpose |
| --- | --- | --- |
| `TWITTER_BEARER_TOKEN` | optional | Direct Twitter API access in `desearch/services/twitter_api_wrapper.py`; not required if the miner relies on Apify. |
| `WALLET_NAME` | no | Default wallet name for the axon; default `miner`. `--wallet.name` overrides. |
| `WALLET_HOTKEY` | no | Default hotkey; default `default`. `--wallet.hotkey` overrides. |
| `SUBTENSOR_NETWORK` | no | Chain/network selector such as `finney`, `test`, or a custom endpoint; default `finney`. `--subtensor.network` overrides. |
| `NETUID` | no | Subnet UID; default `22`. `--netuid` overrides. |
| `AXON_PORT` | no | Axon port; default `8098` in `neurons/miners/.env.template`. `--axon.port` overrides. |

### Miner `.env` example

```dotenv
WALLET_NAME=miner
WALLET_HOTKEY=default
SUBTENSOR_NETWORK=finney
NETUID=22
AXON_PORT=8098
OPENAI_API_KEY=***
SCRAPINGDOG_API_KEY=***
APIFY_API_KEY=***
# TWITTER_BEARER_TOKEN=***
```
