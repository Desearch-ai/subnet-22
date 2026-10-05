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
btcli wallet new-coldkey --wallet miner
btcli wallet new-hotkey --wallet miner --wallet-hotkey default
btcli subnets register --netuid 22 --wallet miner --wallet-hotkey default --network finney
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
| `TASK_API_URL` | `https://api-22.desearch.ai` | the task API |
| `PROXY_URLS` | none | comma-separated `http://user:pass@host:port`, rotated per request |
| `CRAWL_CONCURRENCY` | 32 | pages fetched at once |
| `CRAWL_CONCURRENCY_PER_DOMAIN` | 8 | pages fetched at once from one domain, so a site does not block you |
| `CRAWL_FIRST_TIMEOUT` | 10 | seconds for a page's first try |
| `CRAWL_TIMEOUT` | 30 | seconds for each later try |
| `CRAWL_ATTEMPTS` | 3 | tries per page from your own addresses before ScrapingDog or an error row |
| `CRAWL_USER_AGENT` | a desktop Chrome string | |
| `MAX_TASKS` | 4 | most tasks held at once; the miner also claims only what its measured pace can finish, within its budget |
| `EXTRACTION_THREADS` | 4 | pages turned into text at once; each holds a whole page in memory |
| `SCRAPINGDOG_API_KEY` | none | optional: a page your address cannot load (refused, blocked, timed out) is retried through ScrapingDog |
| `SCRAPINGDOG_CONCURRENCY` | 8 | ScrapingDog requests at once; they wait outside the crawl slots, so your own fetches keep going |
| `RECEIPTS_FILE` | none | file each signed receipt is appended to, as proof of what you were served |

A task must be uploaded within 3 minutes of claiming it. The miner tries every page once with a
short timeout, then retries slow and failed pages behind the untried ones. It stops 30 seconds before
the end (longer if your uploads have been slow) to write and upload the file; unfinished pages go in
as timed out, and the log says how many, so you know to raise `CRAWL_CONCURRENCY` or lower
`MAX_TASKS`.

A page refused for its address (403, 408, 429) or served a challenge is retried through the next
proxy, or without proxies straight through ScrapingDog; 404 and 410 are final. Each proxied request
opens a fresh connection, so a rotating gateway gives a new exit address each time. Private addresses
are refused only on direct connections, so use proxies that reach the public internet only.

## 4. Run

From the repository root:

```bash
pm2 start python3 --name desearch_miner -- -m neurons.miners.miner
```

## Test your miner locally

Before registering, run your miner against a local task API and validator that use the same code as
mainnet, and see exactly how your uploads are scored:
[Test your miner locally](./test-your-miner.md).

## Embedding (not open yet)

Embed tasks open when Desearch's own embedding model ships, and run on your own GPU.
[Embedding tasks](./embedding-tasks.md) describes what you will receive, the model to run and what
to return.

## How you earn

Your share is your paid pages over the last 24 hours, compared with every other miner's.
Validators check a drawn share of the tasks you upload, and every one until your hotkey has passed
10 checks; a checked task pays at the rate its sample matched, and the others are paid on the counts
your miner reports. A failed check takes back what you were paid since your last passed one.
[Emission](./emission.md) explains the rules and what raises your share.

## Monitor

```bash
pm2 logs desearch_miner
curl -s https://api-22.desearch.ai/v1/miners/<hotkey>
curl -s "https://api-22.desearch.ai/v1/tasks?miner=<hotkey>"
```

The first shows your budget, tasks in progress, uploads waiting for a verdict, coverage and
pass/fail counts; the second your checked tasks; `/v1/tasks/<task_id>` what validators found per URL.
