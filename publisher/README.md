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

The planner embeds each published page of v2 extraction (recipe `page-v2`) as one page text, its title,
description and the start of its text, and a long page also as up to 32 passages split at its headings,
never inside a table row, a list item or a code block. A v1 page keeps the vectors it has.

## Planner

The planner reads the change log in log order, fetching `PLANNER_READS` (16) change files ahead while it plans,
and cuts inputs of up to 1,000 pages or 3M characters for the task API, writing up to `PLANNER_WRITES` (8) of
them at once to the temporary bucket. Inputs are saved in the order cut, each once written: the oldest when
`PLANNER_WRITES` are unsaved and another is cut, and all of them when a lane holds `PLANNER_MAX_PENDING` inputs
or the pass ends. A failed write or save fails the pass, and the same input is tried again before any later one
is saved. Saved inputs go to the API while it has room, the fresh lane's first, then oldest first, no faster
than `PLANNER_PAGES_PER_HOUR` and, with the fresh lane on, `PLANNER_FRESH_PER_HOUR` together allow. A starting
planner holds no budget but the fresh lane's two minutes, and budget left unspent is kept for two minutes at
most, so neither a restart nor a stretch without room plans or hands over a backlog at once. A restart reads
the log again from before the oldest page not in a saved input, and plans again only pages no saved input
holds. A starting planner waits up to three minutes for the one it replaces to let go of its state. When the
API is full, slow to answer (2 s or more) or does not answer, inputs wait, and the planner asks for room at the
end of each pass and at most every 30 s in between; after a restart it asks before reading the log.

After a pass, at most once a minute, the planner logs its place in the log, its counts since it started, and the
seconds it spent since then reading the log and planning waiting pages, waiting on change files for each, waiting
on input writes and writing its state.

- **What it plans** (`PLANNER_SET_FILE`): only pages the v1 set takes: a news site's pages dated on or
  after its `since`, by `published` or a date in the URL path, and, when the file's `undated` is `article`,
  its pages with no date typed `article`; a whole site's pages, or those under its `paths`; a bulk source's
  crawled pages. Pages are also English, at least `PLANNER_MIN_CHARS` long and off
  the exclusion list. With `PLANNER_SET_FILE` set, the planner does not start without the set. Each time it reads
  the set it logs the set's `built` time, or the time its file was written when the set has none or its `built`
  cannot be parsed, and it warns then and hourly while the set is more than two days old; the same build read again
  keeps the hourly timer.
- **Classes:** while the budget or the API's room is short, a site's pages go by its class in the set, best first:
  a site a benchmark cites (any word in its `why` other than `wikidata`, `news-list`, `news-major`, `news`, `judge`,
  `academic`, `public`, `docs`, `evergreen` and `rank`), a news site with no `since`, a docs site, a reference site,
  a news site with a `since`, then every other site. A page from the log takes a place, budget and room both, only
  beyond what the log's better classes asked for in its last passes that read change files and have not asked for
  yet this pass, and beyond the waiting pages of better classes in the change files the re-plan reads next. Waiting
  and backfilled pages leave what the log asked for of their class and better. Without a set, every page is of the
  last class. The report line counts the log's pages that asked for a place, and the pages planned outside the
  fresh lane, by class.
- **Fresh lane:** pages seen for the first time and dated within `PLANNER_FRESH_S` (48 h) get their own inputs,
  written within `PLANNER_FRESH_FLUSH_S` (60 s), handed over ahead of other waiting inputs, outside the hourly
  budget (`PLANNER_PAGES_PER_HOUR`) and within their own (`PLANNER_FRESH_PER_HOUR`). Every other page shares the
  budget. A page the log removes is forgotten, so it is embedded again if it comes back; until a row of it is
  planned or waits, rows from before the removal are passed over.
- **Waiting pages:** a page refused for budget or room, or given back after it was planned, waits with the change
  file that holds it, under its site's class. Once `PLANNER_MAX_WAITING` pages wait, a page waits only in place of
  the page of the worst class worse than its own that has waited longest, which is left to the backfill
  (`evicted`); otherwise it is not kept. A page is planned from its change file once there is room and the budget
  holds an input's worth of pages (`PLANNER_PAGES_PER_INPUT`, or two minutes of budget when that is less, as the
  budget holds no more, or the class's waiting pages counted at the pass's start when they are fewer) beyond the
  log's share for its class. The re-plan goes class by class, best first, and stops within a file once only that
  share is left, reading the file again next pass. A pass reads up to 64 such change files, oldest first within a
  class, fetching up to `PLANNER_READS` (16) of them ahead while the budget and room would still hold an input's
  worth once the pages waiting in those already fetched were taken. A page taken from them waits until its input is
  saved. A version planned without its change file number is left to the backfill or a re-crawl. An input is pushed
  to the task API once, however often a restart tries it. A starting planner gives a class to each waiting page
  that has none.
- **Backfill** (`PLANNER_BACKFILL=1`): with what budget and room the log and waiting pages leave, it reads
  `current/` shard by shard (`PLANNER_BACKFILL_SHARDS`), a sweep for each class the set holds sites of, best first,
  each taking the pages of that class and better. Each pass re-plans waiting pages first. The backfill starts once
  the budget holds an input's worth beyond the log's share for its sweep's class, but not in a pass where the
  re-plan stopped for budget at that class or a better one; when the re-plan stops at its 64 change files a pass or
  the end of its sweep, the backfill takes what is left, so the two share the budget while pages wait. A backfill pass reads at most `PLANNER_MAX_PENDING` ×
  `PLANNER_PAGES_PER_INPUT` pages, for at most `PLANNER_FRESH_FLUSH_S`, so the log is read between. It plans each
  page's standing version that the set takes and was never embedded at that text, so a site the set marks `out` is
  never read past its small columns, and fetches at most 64 MB of row groups at once. A shard that cannot be read
  is tried again from the place kept `PLANNER_BACKFILL_RETRY_S` (600) later, while the log and waiting pages go on;
  after three failed reads in a row the backfill goes on to the next shard, logs that it skipped it, and counts
  both in its report. It keeps its place and class across restarts, and sweeps again from the best class
  `PLANNER_BACKFILL_EVERY_S` (24 h) after its last shard for the last class, so a page the log passed by is planned
  once `current/` holds it.
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
