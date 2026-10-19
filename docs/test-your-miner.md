# 🧪 Test Your Miner Locally

- [Requirements](#requirements)
- [🌐 Test crawling](#-test-crawling)
- [🧠 Test embedding](#-test-embedding)
- [📊 See it in the browser](#-see-it-in-the-browser)
- [🛑 Stopping](#-stopping)

The sandbox runs a task API, storage and a validator on your own machine, with the same code as
mainnet. Point your miner at it to see how your uploads are scored before you register: what passed,
what did not and why, and what you would be paid. It tests the crawl miner, and in embed mode the embed
miner.

It needs no registration and no TAO: any wallet works, registered or not.

## Requirements

- [Docker](https://docs.docker.com/get-docker/), running. Redis, storage and the task API run in
  containers. The sandbox builds the task API's image from this repository the first time it starts,
  which takes several minutes; later starts reuse it until the task API's code changes.
- Python 3.10 or newer, with the sandbox's requirements (the miner's, and boto3):

  ```bash
  python3 -m pip install -r sandbox/requirements.txt
  ```

- For crawling, a [ScrapingDog](https://www.scrapingdog.com/) API key. The local validator checks
  pages the same way a mainnet validator does, including its ScrapingDog fallback for pages its own
  address cannot load. Set `SCRAPINGDOG_API_KEY` in your shell, or in `neurons/validators/.env` or
  `neurons/miners/.env`.
- For embedding, your model server: vLLM on an NVIDIA GPU, as
  [Miner → Embedding](./miner.md#5-embedding) shows. Embed mode needs no ScrapingDog key.

## 🌐 Test crawling

### 1. Start the sandbox

From the repository root:

```bash
python3 -m sandbox
```

The first start builds the task API, printing Docker's progress as it goes. Then it downloads a
file of URLs, queues them as tasks, starts a validator, and prints the address of the local task
API:

```
Building the task API image desearch-sandbox-task-api:0ae23bbd3af1 with Docker. The first build compiles it from source and takes several minutes; later starts reuse it.
...
serving urls-00008.parquet from desearch/sn22-sandbox-urls
queued 1000 URLs as 10 tasks

Task API: http://127.0.0.1:18080
```

Leave it running.

### 2. Start your crawl miner

In a second terminal, start your miner the way you would on mainnet, pointed at the sandbox:

```bash
TASK_API_URL=http://127.0.0.1:18080 python3 -m neurons.miners.miner
```

`TASK_API_URL` can also go in `neurons/miners/.env`. Add your `PROXY_URLS` as you would on mainnet.
Add `LOG_LEVEL=DEBUG` to see every page as the miner fetches it.

### 3. Read the results

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
  upload:  http://127.0.0.1:18082/sandbox-uploads/uploads/.../5FkPgJN9...-16.parquet
  details: sandbox/runs/20260929-185547/tasks/249008e8de444962.json
```

The sandbox checks every upload, so you see a verdict for each. Each time the validator sets
weights, it lists each miner's uploads in the 24-hour scoring window, what its own checks found and
the share that follows, and every minute the sandbox prints a summary of your throughput and budget:

```
Scoring window, last 24 h:
  crawl 5FkPgJN98Q: 5 uploads, checked 5 (0 failed), paid at 0.97, 486 rows, share 1.000
summary: 2 tasks waiting, 1 uploads being checked
  5FkPgJN9: 5 tasks checked, 5 passed, 500 pages returned (104/min), 486 rows paid (97%), budget 6
```

How the paid rows follow from the checked pages is explained in [Emission](./emission.md).

### Crawl options

| Option | Default | |
| --- | --- | --- |
| `--task-size` | `100` | URLs per task. Use `1000` for mainnet-sized tasks, to check that your miner finishes one within the 3-minute claim. |
| `--urls FILE` | the public dataset | Serve your own URLs: a parquet file with a `url` column, or a text file with one URL per line. |
| `--port` | `18080` | Port of the local task API; Redis and storage use the next two ports. |

By default the URLs come from the public dataset
[desearch/sn22-sandbox-urls](https://huggingface.co/datasets/desearch/sn22-sandbox-urls): about a
million pages from news and general sites, downloaded one file at a time as the queue needs them.

## 🧠 Test embedding

Embed mode hands out [embed tasks](./embedding-tasks.md) made from a fixed set of texts, each stored
with its reference vector from the model. The validator checks your vectors against those reference
vectors, the way a mainnet validator checks them against its own model server, so you see whether
your model server gives the vectors validators expect before you register.

### 1. Start your model server

Start vLLM on your GPU exactly as [Miner → Embedding](./miner.md#5-embedding) shows: that revision,
in float16.

### 2. Start the sandbox in embed mode

```bash
python3 -m sandbox embed
```

It downloads the test set, hands its texts to the task API as embed inputs, and starts a validator.
The validator checks every upload against the set's reference vectors through a small model server
of the sandbox's own, which knows only the set's texts; your model server is never used for
checking.

```
14,226 texts of 3,000 pages, handed out as 30 inputs of up to 100 pages

Task API: http://127.0.0.1:18080
```

### 3. Start your embed miner

In a second terminal, start the embed miner pointed at the sandbox and at your model server:

```bash
TASK_API_URL=http://127.0.0.1:18080 EMBED_API_URL=http://127.0.0.1:8000/v1/embeddings \
  python3 -m neurons.miners.embed
```

Add `EMBED_API_KEY` if your model server needs one.

### 4. Read the results

The sandbox checks every upload, so you see a verdict for each; on mainnet validators check a share of
them, as [Embedding tasks](./embedding-tasks.md#how-it-is-checked) explains. For each upload, the
sandbox prints the validator's verdict, how many of the sampled texts matched and the lowest cosine
similarity among them, the task's size, how long after the claim the upload came (it must come within
5 minutes), a link to the file your miner uploaded and a JSON file with the full result. Each sampled
text that did not reach 0.999, or whose vector is not finite with length 1, is listed:

```
task 211008abb81642ea miner 5CkpVdiz: FAIL (vectors_mismatch), 14/15 sampled texts matched, lowest similarity 0.989792
  15 texts, 31,120 characters, uploaded 0.4s after the claim (limit 300s)
  https://docs.example.org/emission passage 4: similarity 0.989792
  upload:  http://127.0.0.1:18082/sandbox-uploads/uploads/dt=2026-10-14/task=211008abb81642ea/...f16
  details: sandbox/runs/20261014-224232/tasks/211008abb81642ea.json
```

The JSON file names each sampled text's page and position, and holds the text of each one that did
not match, so you can embed it again yourself. Every minute the sandbox prints your miner's
throughput and embed budget:

```
summary: 12 tasks waiting, 0 uploads being checked
  5GWvYCiS: 19 tasks checked, 19 passed, 225 texts (152/min), 477,340 characters (322,984/min), budget 100
```

### Embed options

| Option | Default | |
| --- | --- | --- |
| `--task-size` | `100` | Pages per task. Use `1000` for mainnet-sized tasks, to check that your miner uploads one within 5 minutes. A task also stays within mainnet's 3 million characters. |
| `--embed-set DIR` | the public set | A folder holding a test set in the public set's layout. |
| `--port` | `18080` | Port of the local task API; Redis, storage and the sandbox's model server use the next three ports. |

By default the texts come from the public dataset
[desearch/sn22-sandbox-embed](https://huggingface.co/datasets/desearch/sn22-sandbox-embed):
English Wikipedia articles made into page texts and passages by the same code that makes mainnet
embed tasks, each with its vector from the model in float32. It is downloaded once; the sandbox hands
its texts out again and again, each time as new inputs.

## 📊 See it in the browser

The same results are available as a web page, for both kinds of task. It needs
[Node.js](https://nodejs.org/) 22 or newer. While the sandbox runs, in another terminal:

```bash
cd ui
npm install
npm run dev
```

Open <http://localhost:5173>. It reads the sandbox's task API and shows the tasks waiting and in
progress, every finalized task with how long your miner took, and the validator's vote:

- for a crawl task, each checked page with what your miner uploaded next to what the validator fetched;
- for an embed task, each checked text with its page and its similarity to the reference vector.

Your miner's page shows its budget, results and share for each kind.

If you started the sandbox with another `--port`, set `VITE_TASK_API_URL` to that address in
`ui/.env.local`.

## 🛑 Stopping

Press `Ctrl-C` in the sandbox's terminal. It stops the validator and the task API, removes its
Docker containers, and deletes the uploaded files, so download one from its link while the sandbox
runs if you want to keep it.

Two folders stay on disk, both ignored by git and safe to delete at any time:

- `sandbox/runs/<start time>/`: the JSON result of every checked task, the task API's data, and the
  validator's and task API's logs;
- `sandbox/cache/`: the downloaded URL files and embed test set, reused by the next run.

The task API's Docker image, `desearch-sandbox-task-api`, stays for the next start. Remove it with
`docker image rm` when you no longer need it.
