# Miner Setup

A miner crawls for the network. It claims tasks from the task API, fetches every page in them
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
through the next proxy; 404 and 410 are taken at face value. Every request through a proxy opens a
fresh connection, so a rotating gateway hands out a new exit address each time. The miner refuses
private addresses only on direct connections: a proxy resolves the hostname itself, so use one that
reaches the public internet only.

## 4. Run

From the repository root:

```bash
pm2 start python3 --name desearch_miner -- -m neurons.miners.miner
```

## Embedding (not open yet)

Embed tasks open when Desearch's own embedding model ships, and run on your own GPU.
[Embedding tasks](./embedding-tasks.md) describes what you will receive, the model to run and what
to return.

## How you earn

Your share is your verified pages over the last 24 hours, compared with every other miner's. A
validators re-fetch a sample of each task you upload, and a passing task pays you at the rate that
sample matched. [Emission](./emission.md) explains the rules and what raises your share.

## Monitor

```bash
pm2 logs desearch_miner
curl -s https://task-api.desearch.ai/v1/miners/<hotkey>
curl -s "https://task-api.desearch.ai/v1/tasks?miner=<hotkey>"
```

The first shows your budget, tasks in flight, coverage and pass/fail counts; the second your checked
tasks, and `/v1/tasks/<task_id>` what the validators found for each URL.
