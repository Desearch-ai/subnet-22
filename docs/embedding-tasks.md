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

Every page gives one **page text**: its title, description and the start of its text. A page over 2,000
characters can also give up to 32 **passages**, split at its headings, each starting with the title and the
section's heading. A task may hold page texts alone.

A task holds the texts of up to 1,000 pages and 3 million characters. Upload its vectors within
5 minutes of claiming it.

## How it is checked

When you complete a task, its file must hold exactly one vector per text. A file of any other size
is refused, and you can upload again while your claim lasts.

Validators check a sample of your uploads, picked by a chain block's hash that nobody knew when you
uploaded:
- every upload of a new hotkey until it has passed 10 checks;
- your next 10 uploads after a failed check;
- otherwise a share of your uploads, at least about one an hour.

In a checked task, validators embed 48 of its texts again with the same model, including some of the
longest. Each of your vectors for them must have finite values and length 1, and reach a cosine
similarity of 0.999 with theirs.

**Pay:** a passing task, and every task not picked for a check, is paid by the characters of its
texts. A failed check earns nothing and costs the task's characters, halves your embed budget and is
a strike. It also takes back everything you uploaded since your last passed check, and those vectors
leave the published set. Two failed checks among your last 10 take back your last 24 hours and lock
you out of embed tasks for 48 hours.

Every vector is checked for finite values and length 1 before it is published. One that fails is
never published, and your next 10 uploads are all checked. An upload picked for a check that no
validator could check in time costs nothing and goes to another miner. Embedding keeps its budget and
lockouts separate from crawling; see [Emission](./emission.md).

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
  "upload": {"url": "<upload link>", "key": "uploads/...", "content_type": "application/octet-stream"},
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

**Output:** one file of vectors with no header: for each row of the input, in the input's order, its
vector normalized to length 1, as 2,560 little-endian float16 values (5,120 bytes). The file is
exactly the number of texts × 5,120 bytes.

Upload it to `upload.url` with the given content type, then call `POST /v1/tasks/{task_id}/complete`.
