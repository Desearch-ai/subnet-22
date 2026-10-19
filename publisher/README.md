# Publisher

Writes uploads the task API hands over to the pages bucket: crawl pages to the `changes` feed, compacted
into `current/`, and embed vectors to one feed per model. The planner (`publisher plan`) turns published
pages into embed inputs.

## Vectors

```
vectors/<model>/
  log/dt=<day>/<id>.parquet   one file per embed task, or a withdrawal entry
  seq/<number>.json           {"key": <file>, "rows": <rows>}, numbered in the order written
  latest.json                 {"seq": <newest number>}
```

Every file has the same columns. A task's file has a row per text: `key` (the page), `url`, `content_sha1`,
`kind` (`page` or `passage`), `index`, `vector` (float16, little-endian), `task_id`, `miner`,
`published_at`.

A withdrawal entry has a row per task taken back after its vectors were published: `kind` is
`task_withdrawn`, with `task_id`, `miner` and `published_at` set, `key`, `url` and `content_sha1` empty,
`index` 0 and `vector` empty. A reader drops every vector row of that task, as a `task_withdrawn` row in
`changes` drops a task's pages.

A vector that is not finite or whose length is not 1 (within 0.01) is never published. Its page is left out
of the task's file and goes back to the planner through `embed:dropped`, and the task is listed on
`embed:malformed` for the task API. A task taken back sends its pages back to the planner the same way.
