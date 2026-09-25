# Embedding Tasks

> **Not open yet.** Embedding opens when Desearch's own embedding model ships. Until then it pays
> nothing, and a miner asking for an embed task is told `KIND_CLOSED`. This page describes what the
> tasks will be, so you can prepare.

Embed tasks turn the pages miners crawl into the vectors the search index is built from. Each task is
a batch of texts cut from newly crawled pages, with the name of the model to run them through. You
run the model on your own GPU and return one vector per text; a validator recomputes a sample and
checks yours match.

## What you receive

Claim a task with `POST /v1/tasks/claim` and the body `{"kind": "embed"}`:

```json
{
  "task_id": "9f1c2a7b3e4d5f60",
  "kind": "embed",
  "model": "qwen3-embedding-8b",
  "texts": 3912,
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
| `page_key`, `url`, `content_sha1` | the page and the version of it the text was cut from |
| `kind` | `head`, `full` or `chunk` |
| `index` | the passage number for a `chunk`, otherwise 0 |
| `text` | exactly what to embed |

Every page gives a `head` (its title and first 1,500 characters), a `full` text (its title and first
8,000 characters) and one `chunk` per passage. A task holds up to 500 pages, usually a few thousand
texts.

## The model

A task names the model to run. Until Desearch's model ships, tasks name the stand-in used for
testing:

| | |
| --- | --- |
| Name | `qwen3-embedding-8b` |
| Weights | [`Qwen/Qwen3-Embedding-8B`](https://huggingface.co/Qwen/Qwen3-Embedding-8B), revision `1d8ad4ca9b3dd8059ad90a75d4983776a23d44af` |
| Vector size | 4096 |

New models are added to [`desearch/embedding.py`](../desearch/embedding.py) under their own names.
Your miner only takes tasks for the model it runs.

To match the validator's vectors:

- embed each text exactly as given: no prompt or instruction in front, nothing added, nothing cut;
- use the model's own pooling and normalization, as vLLM and Sentence Transformers do by default;
- half precision (float16 or bfloat16) is fine.

## What you return

A Parquet file with one row per `text_id`:

| Column | |
| --- | --- |
| `text_id` | as given |
| `vector` | the vector, length 1, as little-endian float16 bytes (4096 × 2 = 8,192 bytes) |

Upload it to `upload.url` with the given content type, then call `POST /v1/tasks/{task_id}/complete`.

## How it is checked

1. Every `text_id` must have exactly one vector of the right size, with finite values and length 1.
2. 20 texts, chosen after your upload, are embedded again by the validator with the same model. Each
   must reach a cosine similarity of 0.99 with yours. The same model on different hardware scores
   above 0.999; a different model scores near 0.

A task that fails either check earns nothing and counts against your embed budget. Embedding keeps its
budget and lockouts separate from crawling. A passing task is paid by the characters of its
texts; see [Emission](./emission.md).

## Running a miner

Serve the model on your GPU with vLLM, then start the embed miner next to your crawl miner, under
the same hotkey. The 8B model needs about 16 GB of GPU memory for its weights, so a 24 GB card or
larger:

```bash
vllm serve Qwen/Qwen3-Embedding-8B --revision 1d8ad4ca9b3dd8059ad90a75d4983776a23d44af --runner pooling
pm2 start python3 --name desearch_embed_miner -- -m neurons.miners.embed
```

| Variable | Default | |
| --- | --- | --- |
| `EMBED_MODEL` | `qwen3-embedding-8b` | the model you run |
| `EMBED_API_URL` | `http://127.0.0.1:8000/v1/embeddings` | your model server |
| `EMBED_API_KEY` | none | if your server requires one |
