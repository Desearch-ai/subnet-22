# desearch-bot

The DesearchBot crawl loop. For every domain it owns, it reads robots.txt and the sitemaps on a
schedule, keeps every URL they list, and reports each visit to the registry. It also sends new and
changed pages to the subnet's task API and learns from the API what became of them. One process uses
every core.

- **Stores**: 256 RocksDB bucket stores holding domains, sitemaps and URLs, what each sitemap listed
  when last read, and the pages waiting to go to the task API.
- **Registry**: the `bot.domains` table in Postgres, which records each domain's state. The schema is
  [`schema.sql`](./schema.sql).
- **Rules**: robots.txt and Crawl-delay, one request a second per host, sitemap schedules, exclusions,
  and Web Bot Auth request signatures.

The domain list is published as the Hugging Face dataset
[`desearch/subnet-22`](https://huggingface.co/datasets/desearch/subnet-22); its card is
[`dataset_card.md`](./dataset_card.md).

## Build

```bash
cargo build --release
```

RocksDB is compiled from source, which needs `clang` and a C++ toolchain.

## Run

```bash
python tools/export_langid.py data/langid.bin   # once: the language model homepages are judged with
MALLOC_ARENA_MAX=2 desearch-bot run --buckets-dir /var/lib/desearch-bot/buckets --data-dir data
```

| Option | Default | |
| --- | --- | --- |
| `--concurrency` | 840 | visits in flight at once |
| `--buckets` | `0-255` | buckets to crawl, as ranges such as `0-13,20` |
| `--buckets-dir` | `/var/lib/desearch-bot/buckets` | where the stores live |
| `--data-dir` | `data` | where `public_suffix_list.dat` and `langid.bin` live |
| `--no-registry` | off | crawl without Postgres |
| `--discovery-share` | 30 | percent of the visit slots kept for domains never checked |
| `--backlog-share` | 20 | percent kept for sitemaps found but not read yet; re-checks of known sites get the rest |
| `--task-api` | none | the task API to fill, such as `https://api-22.desearch.ai` |
| `--outcomes-url` | none | where the task API publishes what became of each URL, such as `https://r2.desearch.ai` |
| `--per-domain-hourly` | 2000 | URLs one domain may send in an hour |
| `--new-share` | 50 | percent of each domain's sends kept for pages never sent |
| `--retry-share` | 10 | percent kept for retries and scheduled re-crawls; changed pages get the rest |
| `--recrawl-days` | 7 | days after a crawl that a page without a lastmod goes out again |
| `--backfill-ready` | off | walk every store once to queue pages found before the queue existed |
| `--domains` | none | a JSON list of the domains whose pages are queued, as names or `{host, rank}`, read again when it changes; others are crawled but not queued |
| `--backfill-sent` | none | pages the Python feeder sent, as tab-separated host, path, lastmod and time, marked sent first |
| `--duration` | none | stop after this many seconds |

`desearch-bot run --help` lists the memory and disk limits.

| Variable | |
| --- | --- |
| `DESEARCH_DB` | the registry's Postgres DSN |
| `DESEARCH_SIGNING_KEY_FILE` | the Ed25519 key requests are signed with |
| `FEEDER_KEY_URI` | the task API admin hotkey, as a Substrate secret URI, needed with `--task-api` |
| `MALLOC_ARENA_MAX=2` | stops glibc from holding on to memory RocksDB has freed |

Visits share their slots between re-checks of known sites, sitemap backlogs and domains never
checked. Each kind can count on its share, slots one leaves unused go to the others, and due domains
go best ranked first. A sitemap read again looks up only the entries it did not list the same way
last time, from a compact fingerprint set kept per sitemap.

## Feeding the task API

A page joins its domain's queue when a sitemap lists it for the first time, or lists it with a
lastmod past the one it was sent with. Listing, search and file pages never join. A sitemap that
moves more than 2% of its pages, and more than 50, at once is re-stamping dates, and only its 10
newest go.

Each domain's queue has three lanes that send at the same time:

- **new**: pages never sent, newest found first, so a post published today goes on the next pass;
- **changed**: pages whose lastmod moved, newest change first;
- **retries**: failed pages coming back after 1, 6 and 24 hours, and pages without a lastmod due for
  a re-crawl.

Each lane gets its share of what the domain sends (`--new-share`, `--retry-share`, and the rest for
changed pages), kept even across passes, and what one lane leaves unused goes to the others.

Every 15 seconds the bot asks `GET /v1/room` how many tasks the API can take, and sends that many
times 1,000 URLs, a few from each domain in turn, best ranked first, within `--per-domain-hourly`. A
batch carries a hash of its content, so a batch sent again after a timeout is queued once, and a
batch in flight survives a restart. It then reads `outcomes/seq/<n>.json` in order: a published or
unchanged page waits for its next lastmod change, and a failed or dropped one becomes a retry. A
status line every minute shows the queue, what was sent in each lane, the room and the outcome feed.

Pages found before the queue existed join it once through `--backfill-ready`. Pages the Python
feeder already sent are marked first, from its state:

```bash
sqlite3 -separator $'\t' sent.db "SELECT host, path, lastmod, CAST(at AS INTEGER) FROM sent" > sent.tsv
desearch-bot run --backfill-ready --backfill-sent sent.tsv ...
```

Run one bot with `--task-api` at a time.

SIGTERM and SIGINT stop the loop; visits still running after 30 seconds are dropped and simply fall
due again.

## Tests

```bash
cargo test --release
```

The parity tests replay fixed vectors for URL keys, joins, dates, signatures and language, byte for
byte. `tests/registry.rs` needs `initdb` and `postgres` on the PATH, `tests/world.rs` crawls a
small local web end to end, and `tests/dispatch.rs` runs the queue, its lanes, the outcome feed and
the dispatcher against a fake task API.
