# Miner Setup

A miner crawls for the network. It leases tasks from the task API, fetches every page in them
through its own proxies, extracts the text and uploads it. Validators check the uploads, so a miner
needs no open port.

The miner is [`neurons/miners/`](../neurons/miners/). It fetches and extracts pages with the shared
[`desearch`](../desearch/) package, the same code validators check your pages with.

## Requirements

- Python 3.10 or newer, and [PM2](https://pm2.io/docs/runtime/guide/installation/)
- A hotkey registered on netuid 22 (netuid 41 on testnet)
- Proxies: sites refuse repeated requests from one address, and a page a validator can load but you
  could not is not paid

## 1. Install

```bash
git clone https://github.com/Desearch-ai/subnet-22.git
cd subnet-22
python3 -m pip install -r requirements.txt
python3 -m pip install -e .
```

## 2. Register a hotkey

```bash
btcli wallet new_coldkey --wallet.name miner
btcli wallet new_hotkey --wallet.name miner --wallet.hotkey default
btcli subnet register --netuid 22 --wallet.name miner --wallet.hotkey default --subtensor.network finney
```

## 3. Configure

```bash
cp neurons/miners/.env.template neurons/miners/.env
```

The miner reads `neurons/miners/.env` on start; variables already set in the shell win.

| Variable | Default | |
| --- | --- | --- |
| `WALLET_NAME` | `default` | wallet holding the registered hotkey |
| `WALLET_HOTKEY` | `default` | hotkey the miner signs its requests with |
| `WALLET_PATH` | `~/.bittensor/wallets` | |
| `TASK_API_URL` | `https://task-api.desearch.ai` | the task API |
| `PROXY_URLS` | none | comma-separated `http://user:pass@host:port`, rotated per request |
| `CRAWL_CONCURRENCY` | 32 | pages fetched at once |
| `CRAWL_CONCURRENCY_PER_DOMAIN` | 8 | pages fetched at once from one domain, so a site does not block you |
| `CRAWL_TIMEOUT` | 30 | seconds per page |
| `CRAWL_USER_AGENT` | a desktop Chrome string | |
| `MAX_TASKS` | 4 | tasks held at once, capped by the budget the API grants |
| `EXTRACTION_THREADS` | 4 | pages turned into text at once; each holds a whole page in memory |
| `SCRAPINGDOG_API_KEY` | none | optional: a page your address cannot load (refused, blocked, timed out) is retried through ScrapingDog |
| `SCRAPINGDOG_CONCURRENCY` | 8 | ScrapingDog requests at once; they wait outside the crawl slots, so your own fetches keep going |
| `RECEIPTS_FILE` | none | file each signed receipt is appended to, as proof of what you were served |

A page refused for the address it came from (403, 408, 429) or served a challenge is fetched again
through the next proxy; 404 and 410 are taken at face value.

## 4. Run

From the repository root:

```bash
pm2 start python3 --name desearch_miner -- -m neurons.miners.miner
```

## Embedding

Embed tasks need no crawling setup. Each one is a file of texts to turn into vectors with the model
the task names (today `qwen3-embedding-8b`, 4096 numbers per text); the vectors must match what a
validator gets from the same model. The embed miner calls any OpenAI-compatible `/embeddings`
endpoint that serves the model:

| Variable | Default | |
| --- | --- | --- |
| `EMBED_MODEL` | `qwen3-embedding-8b` | the model this miner runs; tasks for another are handed back |
| `EMBED_API_URL` | OpenRouter's `/embeddings` | any OpenAI-compatible endpoint |
| `EMBED_API_KEY` | none | required |
| `EMBED_PROVIDERS` | none | comma-separated OpenRouter providers to pin, e.g. `DeepInfra` |

```bash
pm2 start python3 --name desearch_embed_miner -- -m neurons.miners.embed
```

It runs next to the crawl miner under the same hotkey, with its own budget, lockout and pool.

## How you earn

- The API grants your hotkey a budget of tasks it may hold, from lease until verdict. It starts at 1,
  grows by one for every task credited for at least 85% of its URLs, and halves when a task fails,
  a lease expires or a task is abandoned.
- Two failed tasks within 24 hours, if they are at least 5% of your verdicts, lock your hotkey out of
  new tasks for 12 hours. You are never given
  a task you held before, and a refused lease says how long to wait; the miner waits that long.
- A validator re-fetches a sample of each upload. A passing task is credited for its pages at the
  rate its sample matched.
- Your share is your credited pages over the last 24 hours, if you returned at least 85% of the URLs
  assigned to you in that time. Half of the subnet's emission is split by these shares.

The full rules are in [How Subnet 22 works](./how-it-works.md).

## Monitor

```bash
pm2 logs desearch_miner
curl -s https://task-api.desearch.ai/v1/miners/<hotkey>
curl -s "https://task-api.desearch.ai/v1/tasks?miner=<hotkey>"
```

The first shows your budget, tasks in flight, coverage and verdicts; the second your scored tasks,
and `/v1/tasks/<task_id>` what the validator found for each URL.
