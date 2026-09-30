# Test Your Miner Locally

The sandbox runs a task API, storage and a validator on your own machine, with the same code as
mainnet. Point your miner at it to see how your uploads are scored before you register: which pages
passed, which did not and why, and how many rows you would be paid.

It needs no registration and no TAO: any wallet works, registered or not.

## Requirements

- [Docker](https://docs.docker.com/get-docker/), running
- Python 3.10 or newer, with the requirements of both the miner and the task API:

  ```bash
  python3 -m pip install -r requirements.txt -r task-api/requirements.txt
  ```

- A [ScrapingDog](https://www.scrapingdog.com/) API key. The local validator checks pages the same
  way a mainnet validator does, including its ScrapingDog fallback for pages its own address cannot
  load. Set `SCRAPINGDOG_API_KEY` in your shell, or in `neurons/validators/.env` or
  `neurons/miners/.env`.

## 1. Start the sandbox

From the repository root:

```bash
python3 -m sandbox
```

It downloads a file of URLs, queues them as tasks, starts a validator, and prints the address of
the local task API:

```
serving urls-00008.parquet from desearch/sn22-sandbox-urls
queued 1000 URLs as 10 tasks

Task API: http://127.0.0.1:18080
```

Leave it running.

## 2. Start your miner

In a second terminal, start your miner the way you would on mainnet, pointed at the sandbox:

```bash
TASK_API_URL=http://127.0.0.1:18080 python3 -m neurons.miners.miner
```

`TASK_API_URL` can also go in `neurons/miners/.env`. Add your `PROXY_URLS` as you would on mainnet.
Add `LOG_LEVEL=DEBUG` to see every page as the miner fetches it.

## 3. Read the results

**In the miner's terminal**, one line per task and a summary every minute:

```
task d9f7d2a835b447b6: 100 urls, 97 ok, 3 errors (blocked=3), 69.7s, 4132221 bytes uploaded
last 60s: 1 tasks, 100 pages (97 ok), 1.7 pages/s; since start: 1 tasks, 100 pages (97 ok), 0.8 pages/s
```

**In the sandbox's terminal**, the validator's verdict on each upload, every sampled page that did
not check out and why, a link to the exact file your miner uploaded, and a JSON file with the result
for every URL of the task:

```
task 249008e8de444962 miner 5FkPgJN9: PASS (ok), paid 88 of 100 rows, 7 matched / 1 mismatched / 0 unverifiable
  http://qctimes.com/ads/vehicle/car/pdfdisplayad_c1f5fefb.html: mismatched, the texts differ: score 0.00, under the 0.90 threshold
  upload:  http://127.0.0.1:18082/sandbox-uploads/submitted/.../5FkPgJN9...-16-8c8f8a7a.parquet
  details: sandbox/runs/20260929-185547/tasks/249008e8de444962.json
```

Each time the validator sets weights, it lists what each miner did in the 24-hour scoring window,
and every minute the sandbox prints a summary of your throughput and budget:

```
Scoring window, last 24 h:
  crawl 5FkPgJN98Q: 5 tasks (5 passed), 500 of 500 URLs returned, 486 paid, share 1.000
summary: 2 tasks waiting, 1 uploads being checked
  5FkPgJN9: 5 tasks checked, 5 passed, 500 pages returned (104/min), 486 rows paid (97%), budget 6
```

How the paid rows follow from the checked pages is explained in [Emission](./emission.md).

## 4. See it in the browser

The same results are available as a web page. It needs [Node.js](https://nodejs.org/) 22 or newer.
In a third terminal:

```bash
cd ui
npm install
npm run dev
```

Open <http://localhost:5173>. It reads the sandbox's task API and shows the tasks waiting and in
progress, every finalized task with how long your miner took to crawl it, the validator's vote, and
for each checked page what your miner uploaded next to what the validator fetched. Your miner's
page shows its budget, coverage and the rows that counted.

If you started the sandbox with another `--port`, set `VITE_TASK_API_URL` to that address in
`ui/.env.local`.

## Options

| Option | Default | |
| --- | --- | --- |
| `--task-size` | `100` | URLs per task. Use `1000` for mainnet-sized tasks, to check that your miner finishes one well within the 15-minute claim. |
| `--urls FILE` | the public dataset | Serve your own URLs: a parquet file with a `url` column, or a text file with one URL per line. |
| `--port` | `18080` | Port of the local task API; storage uses the next two ports. |

By default the URLs come from the public dataset
[desearch/sn22-sandbox-urls](https://huggingface.co/datasets/desearch/sn22-sandbox-urls): about a
million pages from news and general sites, downloaded one file at a time as the queue needs them.

## Stopping

Press `Ctrl-C` in the sandbox's terminal. It stops the validator and the task API, removes its
Docker containers, and deletes the uploaded files, so download one from its link while the sandbox
runs if you want to keep it.

Two folders stay on disk, both ignored by git and safe to delete at any time:

- `sandbox/runs/<start time>/`: the JSON result of every checked task, and the validator's and task
  API's logs;
- `sandbox/cache/`: the downloaded URL files, reused by the next run.
