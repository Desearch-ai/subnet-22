# ⛏️ Miner

- [📋 What a miner does](#-what-a-miner-does)
  - [💰 How you earn](#-how-you-earn)
- [🔧 Setup](#-setup)
  - [Requirements](#requirements)
  - [1. Install](#1-install)
  - [2. Register a hotkey](#2-register-a-hotkey)
  - [3. Configure](#3-configure)
  - [4. Run](#4-run)
  - [🧪 Test your miner locally](#-test-your-miner-locally)
  - [📈 Monitor](#-monitor)

## 📋 What a miner does

Subnet 22 builds Desearch's web index. Miners do its work in two kinds of tasks, each paid from its own
part of the emission: 30% for crawling, 20% for embedding, and 50% burned. A part no miner earns is
burned too.

- **Crawl tasks:**
  - A task is up to 1,000 URLs from the task API.
  - The miner fetches every page through its own proxies, extracts the text with the shared
    [`desearch`](../desearch/) package and uploads the result within 3 minutes. Validators check pages
    with that same package.
  - Validators re-fetch the pages of a drawn share of uploads and compare them with yours.
- **Embed tasks** (not open yet):
  - A task is a batch of page texts, the texts of up to 1,000 pages.
  - The miner runs Desearch's embedding model on its own GPU and uploads one vector per text within
    5 minutes.
  - Validators embed a sample of the texts again and compare.
  - [Embedding tasks](./embedding-tasks.md) describes the model, the files and the check.

The crawl miner (`neurons.miners.miner`) and the embed miner (`neurons.miners.embed`) run under the same
hotkey, on one machine or two, and either can run alone. Each claims only its own kind of task. A miner
pulls its tasks and uploads its files, so it needs no open port.

### 💰 How you earn

- **Crawling:**
  - Your share is your paid pages over the last 24 hours, compared with every other miner's.
  - Validators check a drawn share of your uploads, and every one until your hotkey has passed 10
    checks.
  - A checked task pays at the rate its sample matched; the others are paid on the counts your miner
    reports.
  - A failed check takes back what you were paid since your last passed one.
- **Embedding:**
  - Every upload is checked.
  - A passing task is paid by the characters of the texts it embedded.
  - A failing one earns nothing and is a strike.

Crawling and embedding keep separate budgets and lockouts. [Emission](./emission.md) explains the rules
and what raises your share.

## 🔧 Setup

### Requirements

- Python 3.10 or newer, and [PM2](https://pm2.io/docs/runtime/guide/installation/)
- A hotkey registered on netuid 22 (netuid 41 on testnet)
- For crawling, proxies: sites refuse repeated requests from one address, and a page a validator can
  load but you could not is not paid
- For embedding, an NVIDIA GPU with 16 GB of memory or more (24 GB recommended) running vLLM; see
  [Embedding tasks](./embedding-tasks.md)

### 1. Install

```bash
git clone https://github.com/Desearch-ai/subnet-22.git
cd subnet-22
python3 -m pip install -r requirements.txt
python3 -m pip install -e .
```

### 2. Register a hotkey

```bash
btcli wallet new-coldkey --wallet miner
btcli wallet new-hotkey --wallet miner --wallet-hotkey default
btcli subnets register --netuid 22 --wallet miner --wallet-hotkey default --network finney
```

### 3. Configure

```bash
cp neurons/miners/.env.template neurons/miners/.env
```

The miners read `neurons/miners/.env` on start; variables already set in the shell win.

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
| `MAX_TASKS` | 4 | most tasks held at once; the crawl miner also claims only what its measured pace can finish, within its budget |
| `EXTRACTION_THREADS` | 4 | pages turned into text at once; each holds a whole page in memory |
| `SCRAPINGDOG_API_KEY` | none | optional: a page your address cannot load (refused, blocked, timed out) is retried through ScrapingDog |
| `SCRAPINGDOG_CONCURRENCY` | 8 | ScrapingDog requests at once; they wait outside the crawl slots, so your own fetches keep going |
| `RECEIPTS_FILE` | none | file each signed receipt is appended to, as proof of what you were served |
| `EMBED_MODEL` | `desearch-4b-v1` | the model your embedding server runs |
| `EMBED_API_URL` | `http://127.0.0.1:8000/v1/embeddings` | your embedding server |
| `EMBED_API_KEY` | none | if your embedding server requires one |

**Crawl timing:**
- A crawl task must be uploaded within 3 minutes of claiming it.
- The miner tries every page once with a short timeout, then retries slow and failed pages behind the
  untried ones.
- It stops 30 seconds before the end (longer if your uploads have been slow) to write and upload the
  file. Unfinished pages go in as timed out, and the log says how many, so you know to raise
  `CRAWL_CONCURRENCY` or lower `MAX_TASKS`.

**Proxies:**
- A page refused for its address (403, 408, 429) or served a challenge is retried through the next
  proxy, or without proxies straight through ScrapingDog. 404 and 410 are final.
- Each proxied request opens a fresh connection, so a rotating gateway gives a new exit address each
  time.
- Private addresses are refused only on direct connections, so use proxies that reach the public
  internet only.

### 4. Run

From the repository root, the crawl miner:

```bash
pm2 start python3 --name desearch_miner -- -m neurons.miners.miner
```

The embed miner (once embed tasks open), after starting the model server as
[Embedding tasks](./embedding-tasks.md) shows:

```bash
pm2 start python3 --name desearch_embed_miner -- -m neurons.miners.embed
```

It waits for the server to answer to the model's name before it claims anything.

### 🧪 Test your miner locally

Before registering, run your miner against a local task API and validator that use the same code as
mainnet, and see exactly how your uploads are scored: [Test your miner locally](./test-your-miner.md).

### 📈 Monitor

```bash
pm2 logs desearch_miner
curl -s https://api-22.desearch.ai/v1/miners/<hotkey>
curl -s "https://api-22.desearch.ai/v1/tasks?miner=<hotkey>"
```

- `/v1/miners/<hotkey>` shows, for each kind of task, your budget, tasks in progress, uploads waiting
  for a verdict, coverage and pass/fail counts.
- `/v1/tasks?miner=<hotkey>` lists your checked tasks.
- `/v1/tasks/<task_id>` shows what validators found for each of a task's URLs or texts.
