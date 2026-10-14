# 🧠 Embedding Tasks

Embed tasks turn the pages miners crawl into the vectors the search index is built from. Each task is
a batch of page texts and the model to run them through. You run the model on your own GPU and return
one vector per text; validators embed a sample themselves and check that yours match.

## The model

| | |
| --- | --- |
| Name | `desearch-4b-v1` |
| Weights | [`desearch/Desearch-Embedding-4B`](https://huggingface.co/desearch/Desearch-Embedding-4B/blob/v1/README.md), tag `v1`, revision `12e5c6910d49dd5534e8a77a6fa7d8a057bd534a` |
| Precision | float16 |
| Vector size | 2560 |
| Input limit | 2,048 tokens |

Serve it with [vLLM](https://docs.vllm.ai) on an NVIDIA GPU with 16 GB of memory or more:

```bash
vllm serve desearch/Desearch-Embedding-4B \
  --revision 12e5c6910d49dd5534e8a77a6fa7d8a057bd534a \
  --served-model-name desearch-4b-v1 \
  --runner pooling --dtype float16 \
  --no-enable-chunked-prefill --max-num-batched-tokens 16384 \
  --port 8000
```

vLLM downloads the weights (about 8 GB) from Hugging Face the first time it starts; no account or token
is needed. To fetch them ahead of time:

```bash
hf download desearch/Desearch-Embedding-4B --revision 12e5c6910d49dd5534e8a77a6fa7d8a057bd534a
```

Run it as given: that revision, in float16. bfloat16, 8-bit or 4-bit weights, another revision or a
shorter input limit give different vectors, and those fail the check. Models are listed in
[`desearch/embedding.py`](../desearch/embedding.py) under their names. A task names its model, and
your miner claims only tasks for the model it runs.

## Running a miner

Start the embed miner once the model server is up. It runs under the same hotkey as your crawl
miner, next to it or on another machine, and either can run alone:

```bash
pm2 start python3 --name desearch_embed_miner -- -m neurons.miners.embed
```

| Variable | Default | |
| --- | --- | --- |
| `EMBED_MODEL` | `desearch-4b-v1` | the model your server runs |
| `EMBED_API_URL` | `http://127.0.0.1:8000/v1/embeddings` | your model server |
| `EMBED_API_KEY` | none | if your server requires one |
| `MAX_TASKS` | 4 | most tasks held at once; they are embedded one after another, and another is claimed only when the measured pace still finishes every held one in time |

The miner waits for the server to answer to the model's name before it claims anything, and stops
with an error when the server runs a different model.

## What you receive

Claim with `POST /v1/tasks/claim` and the body `{"kind": "embed", "models": ["desearch-4b-v1"]}`. A
claim that does not list the tasks' model is refused with `MODEL_MISMATCH`, at no cost. Each task in
the answer's `tasks` list looks like this:

```json
{
  "task_id": "9f1c2a7b3e4d5f60",
  "kind": "embed",
  "model": "desearch-4b-v1",
  "texts": 1000,
  "urls": ["https://example.com/story", "..."],
  "input": {"url": "<download link>", "sha256": "5e3b...c1"},
  "upload": {"url": "<upload link>", "key": "uploads/...", "content_type": "application/vnd.apache.parquet"},
  "expires_at": 1790410000.0
}
```

`input.url` is a Parquet file of the texts. Check it against `input.sha256` before you use it. Each
row holds one text:

| Column | |
| --- | --- |
| `text_id` | what your vector is filed under |
| `page_key`, `url`, `content_sha1` | the page and the version of it the text was taken from |
| `kind`, `index` | `page` and 0 for a page's own text; `passage` and 1, 2, … for an article's passages |
| `text` | exactly what to embed |

Every page gives one page text: its title, description and text. An article over 2,000 characters also
gives up to 16 passages of its paragraphs, split at its section headings, each starting with the title and
the section's heading. A task holds the texts of up to 1,000 pages and 3 million characters.

Embed each text as given, with no prompt in front, and cut it at the model's input limit as vLLM
does with `truncate_prompt_tokens` set to 2048. Upload within 5 minutes of claiming.

## What you return

A Parquet file with one row per `text_id`:

| Column | |
| --- | --- |
| `text_id` | as given |
| `vector` | the vector, length 1, as little-endian float16 bytes (2560 × 2 = 5,120 bytes) |

Upload it to `upload.url` with the given content type, then call `POST /v1/tasks/{task_id}/complete`.

## How it is checked

1. Every `text_id` must have exactly one vector of the right size, with finite values and length 1.
2. 48 texts, chosen after your upload and including some of the longest, are embedded again by
   validators with the same model. Each must reach a cosine similarity of 0.999 with yours.

A task that fails either check earns nothing, halves your embed budget and is a strike. An upload
no validator could check in time costs nothing and goes to another miner. Embedding keeps its budget
and lockouts separate from crawling. A passing task is paid by the characters of its texts; see
[Emission](./emission.md).
