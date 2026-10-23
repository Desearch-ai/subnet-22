# Publisher

Writes uploads the task API hands over to the pages bucket: crawl pages to the `changes` feed, compacted
into `current/`, and embed vectors to one feed per model. The planner (`publisher plan`) turns published
pages into embed inputs.

## Pages

A page row in `changes` and `current/` holds the page's record as the miner extracted it. Fields an older
record does not have are null. Of two rows for one page, the later fetch stands.

## Embed texts

The planner embeds each published page as one page text, its title, description and the start of its text,
and a long page also as up to 32 passages split at its headings, never inside a table row, a list item or a
code block.

## Vectors

```
vectors/<model>/
  log/dt=<day>/<id>.parquet   one file per embed task, or a withdrawal entry
  seq/<number>.json           {"key": <file>, "rows": <rows>}, numbered in the order written
  latest.json                 {"seq": <newest number>}
```

A task's file has a row per text: `key` (the page), `url`, `content_sha1`, `kind` (`page` or `passage`),
`index`, `vector` (float16, little-endian), `task_id`, `miner`, `published_at`, and `recipe`, the planner
recipe that built the text. A file without a `recipe` column holds `page-v1` vectors.

A withdrawal entry has a row per task taken back after its vectors were published: `kind` is
`task_withdrawn`, with `task_id`, `miner` and `published_at` set, `key`, `url`, `content_sha1` and `recipe`
empty, `index` 0 and `vector` empty. A reader drops every vector row of that task, as a `task_withdrawn` row
in `changes` drops a task's pages.

A vector that is not finite or whose length is not 1 (within 0.01) is never published. Its page is left out
of the task's file and goes back to the planner through `embed:dropped`, and the task is listed on
`embed:malformed` for the task API. A task taken back sends its pages back to the planner the same way.
