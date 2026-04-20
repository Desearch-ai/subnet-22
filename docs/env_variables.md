# Environment Variables

Reference for every environment variable consumed by Desearch miners and validators.
Each entry lists who needs it, what it's used for, and where to obtain it.

## Obtaining credentials

- **OpenAI** — https://platform.openai.com/ (API keys)
- **Apify** — https://apify.com/ (Actor-based scraping for Twitter/X verification)
- **ScrapingDog** — https://www.scrapingdog.com/ (web content verification; Standard plan ~$90/mo recommended)
- **SerpAPI** — https://serpapi.com/ (miner web search)
- **Twitter API** — https://developer.twitter.com/en/portal/dashboard (miner direct tweet access)
- **Weights & Biases** — https://wandb.ai/ (validator metrics dashboard)

## Validator variables

| Variable | Required | Purpose |
|----------|----------|---------|
| `OPENAI_API_KEY` | yes | LLM scoring + summary generation. |
| `EXPECTED_ACCESS_KEY` | yes | Gates the public validator API (`neurons/validators/api.py`). Must be ≥16 chars with uppercase, lowercase, digit, and special character. Generate with `python scripts/generate_access_key.py`. |
| `APIFY_API_KEY` | yes | Twitter/X verification via Apify actors. |
| `SCRAPINGDOG_API_KEY` | yes | Web content verification. |
| `WANDB_API_KEY` | yes | Metrics dashboard. Run `wandb login` once after installing. |
| `PORT` | no | Validator API port (default `8005`). |
| `VALIDATOR_SERVICE_PORT` | no | IPC port between API and validator service (default `8006`). |

### Example `.bashrc`

```bash
echo 'export OPENAI_API_KEY="<key>"' >> ~/.bashrc
echo 'export APIFY_API_KEY="<key>"' >> ~/.bashrc
echo 'export SCRAPINGDOG_API_KEY="<key>"' >> ~/.bashrc
echo 'export WANDB_API_KEY="<key>"' >> ~/.bashrc
echo "export EXPECTED_ACCESS_KEY=\"$(python scripts/generate_access_key.py)\"" >> ~/.bashrc
source ~/.bashrc
```

## Miner variables

Miners configure values via `neurons/miners/.env` (copy from `neurons/miners/.env.template`).
CLI args passed to `pm2 start … -- …` take precedence over `.env` values.

| Variable | Required | Purpose |
|----------|----------|---------|
| `OPENAI_API_KEY` | yes | Summary/query generation inside `scraper_miner`. |
| `SERPAPI_API_KEY` | yes | Web search (miners). |
| `APIFY_API_KEY` | yes | Twitter/X scraping. |
| `TWITTER_BEARER_TOKEN` | optional | Direct Twitter API access (`desearch/services/twitter_api_wrapper.py`). Not required if you rely on Apify alone. |
| `WALLET_NAME` | no | Default wallet name used by axon and worker API (default `miner`). `--wallet.name` overrides. |
| `WALLET_HOTKEY` | no | Default hotkey (default `default`). `--wallet.hotkey` overrides. |
| `SUBTENSOR_NETWORK` | no | `finney` / `test` / `local` (default `finney`). `--subtensor.network` overrides. |
| `NETUID` | no | Subnet UID (default `22`). `--netuid` overrides. |
| `AXON_PORT` | no | Axon port (default `14000`). |
| `WORKER_HOST` | no | Worker API bind address (default `0.0.0.0`). |
| `WORKER_PORT` | no | Worker API port (default `8000`). |
