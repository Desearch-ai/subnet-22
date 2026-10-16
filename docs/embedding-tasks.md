# 🧠 Embedding Tasks

Embed tasks turn the pages miners crawl into the vectors the search index is built from. Each task is
a batch of page texts and the model to run them through. You run the model on your own GPU and return
one vector per text; validators embed a sample themselves and check that yours match.

To set up an embed miner, follow [Miner → Embedding](./miner.md#5-embedding). This page describes the
model, the tasks and the check, and the task format for anyone writing their own miner.

## The model

| | |
| --- | --- |
| Name | `desearch-4b-v1` |
| Weights | [`desearch/Desearch-Embedding-4B`](https://huggingface.co/desearch/Desearch-Embedding-4B/blob/v1/README.md), tag `v1`, revision `12e5c6910d49dd5534e8a77a6fa7d8a057bd534a` |
| Precision | float16 |
| Vector size | 2560 |
| Input limit | 2,048 tokens |

Models are listed in [`desearch/embedding.py`](../desearch/embedding.py) under their names. A task names
its model, and your miner claims only tasks for the model it runs.

## What a task holds

Every page gives one **page text**: its title, description and text. An article over 2,000 characters
also gives up to 16 **passages** of its paragraphs, split at its section headings, each starting with the
title and the section's heading.

A task holds the texts of up to 1,000 pages and 3 million characters. Upload its vectors within
5 minutes of claiming it.

## How it is checked

1. Every text must have exactly one vector of the right size, with finite values and length 1.
2. 48 texts, chosen after your upload and including some of the longest, are embedded again by
   validators with the same model. Each must reach a cosine similarity of 0.999 with yours.

A task that fails either check earns nothing, halves your embed budget and is a strike. An upload
no validator could check in time costs nothing and goes to another miner. Embedding keeps its budget
and lockouts separate from crawling. A passing task is paid by the characters of its texts; see
[Emission](./emission.md).

## Task format

The embed miner in this repository does all of this. It is here for anyone writing their own.

**Claim** with `POST /v1/tasks/claim` and the body `{"kind": "embed", "models": ["desearch-4b-v1"]}`. A
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

**Input:** `input.url` is a Parquet file of the texts. Check it against `input.sha256` before you use it.
Each row holds one text:

| Column | |
| --- | --- |
| `text_id` | what your vector is filed under |
| `page_key`, `url`, `content_sha1` | the page and the version of it the text was taken from |
| `kind`, `index` | `page` and 0 for a page text; `passage` and 1, 2, … for an article's passages |
| `text` | exactly what to embed |

Embed each text as given, with no prompt in front, and cut it at the model's input limit as vLLM
does with `truncate_prompt_tokens` set to 2048.

**Output:** a Parquet file with one row per `text_id`:

| Column | |
| --- | --- |
| `text_id` | as given |
| `vector` | the vector, length 1, as little-endian float16 bytes (2560 × 2 = 5,120 bytes) |

Upload it to `upload.url` with the given content type, then call `POST /v1/tasks/{task_id}/complete`.
