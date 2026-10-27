# Publisher

Writes uploads the task API hands over to the pages bucket: crawl pages to the `changes` feed, compacted
into `current/`, and embed vectors to one feed per model. The planner (`publisher plan`) turns published
pages into embed inputs.

## Pages

A page row in `changes` and `current/` holds the page's record as the miner extracted it. Fields an older
record does not have are null. Of two rows for one page, the later fetch stands.

## Compactor

`publisher compact` folds the `changes` feed into `current/` a cycle at a time: it reads the next change
files, writes a delta file to each shard, then combines small deltas and rewrites bases. A shard's base is
rewritten once its deltas hold `COMPACTOR_MERGE_RATIO` of it; otherwise its smallest size of delta is combined
into one of the next size once there are `COMPACTOR_FANOUT` of them. A merge holds every file it reads and the
file it writes in memory. A failed merge leaves its shard as it was for the next cycle, and a retired file
whose delete fails stays listed to be deleted again.

| Setting | Default | |
|---|---|---|
| `COMPACTOR_FILES_PER_CYCLE` | 300 | change files read a cycle |
| `COMPACTOR_MERGES` | 64 | base rewrites a cycle; with no combine budget, combinings share these slots after the base rewrites |
| `COMPACTOR_COMBINES` | 0 | combinings a cycle, run before base rewrites; 0 leaves them the slots of `COMPACTOR_MERGES` that base rewrites leave |
| `COMPACTOR_MERGE_THREADS` | 1 | merges at once; each holds a whole shard in memory |
| `COMPACTOR_THREADS` | 8 | change files read and deltas written at once, and a merge's files fetched and retired files deleted at once |
| `COMPACTOR_MERGE_RATIO` | 1.0 | a base is rewritten once its deltas hold this share of its rows |
| `COMPACTOR_FANOUT` | 12 | deltas of one size combined into one of the next |
| `COMPACTOR_RETIRE_S` | 86400 | seconds a replaced file stays for readers mid-read |
| `COMPACTOR_SPILL_MB` | 1024 | rows held in memory before they spill to `COMPACTOR_WORK` |
| `COMPACTOR_WORK` | `compactor-work` | local folder for spilled rows; a cycle spills about 1 GB per 100 change files |
| `COMPACTOR_EVERY_S` | 900 | once caught up, a cycle starts this long after the last |
| `COMPACTOR_HOLE_WAIT_S` | 600 | a number unwritten this long after a later one is skipped |
| `COMPACTOR_LAG_WARN_S` | 432000 | a warning once the oldest unread change file is this old; the change log keeps ten days |

Each cycle logs one line: the change files read, rows, deltas, rewrites, deletes, how long it took and how old
the oldest unread change file is.

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
