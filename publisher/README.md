# Publisher

Writes uploads the task API hands over to the pages bucket: crawl pages to the `changes` feed, compacted
into `current/`, and embed vectors to one feed per model. The planner (`publisher plan`) turns published
pages into embed inputs.

## Pages

A page row in `changes` and `current/` holds the page's record as the miner extracted it. Fields an older
record does not have are null. Of two rows for one page, the later fetch stands.

## Embed texts

The planner embeds each published page of v2 extraction (recipe `page-v2`) as one page text, its title,
description and the start of its text, and a long page also as up to 32 passages split at its headings,
never inside a table row, a list item or a code block. A v1 page keeps the vectors it has.

## Planner

The planner reads the change log, a few files at once and in log order, and writes inputs of up to 1,000
pages or 3M characters for the task API, handing them over as the API has room.

- **What it plans** (`PLANNER_SET_FILE`): only pages the v1 set takes: a news site's pages dated on or
  after its `since`, by `published` or a date in the URL path; a whole site's pages, or those under its
  `paths`; a bulk source's crawled pages. Pages are also English, at least `PLANNER_MIN_CHARS` long and off
  the exclusion list.
- **Fresh lane:** pages dated within `PLANNER_FRESH_S` (48 h) get their own inputs, handed over first and
  outside the hourly budget (`PLANNER_PAGES_PER_HOUR`). Every other page shares the budget.
- **Backfill** (`PLANNER_BACKFILL=1`): with what budget and room the log leaves, it reads `current/` shard
  by shard (`PLANNER_BACKFILL_SHARDS`) and plans each page's standing version that the set takes and was
  never embedded at that text. It keeps its place across restarts.
- **Page vectors first** (`PLANNER_PAGE_FIRST=1`, off by default): pages outside the fresh lane get their
  page text alone; a page's passages are planned the next time it is seen with the setting off.

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
