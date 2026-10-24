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
| `--outcomes-url` | none | where the task API publishes what became of each URL, such as `https://subnet-22.s3.us-east-va.io.cloud.ovh.us` |
| `--per-domain-hourly` | 2000 | URLs one domain may send in an hour |
| `--new-share` | 50 | percent of each domain's sends kept for pages never sent |
| `--retry-share` | 10 | percent kept for retries and scheduled re-crawls; changed pages get the rest |
| `--recrawl-days` | 7 | days after a send that a page without a lastmod, or a page that failed, goes out again |
| `--recrawl-overrides` | none | a file of `domain days` lines, such as `bbc.co.uk 1`, giving listed domains and their subdomains their own interval |
| `--gone-days` | 30 | days a page that failed on two sends in a row waits before it goes out again, unless its lastmod moves |
| `--dropped-per-hour` | 200000 | pages of lost or taken-back tasks sent again each hour; more are spread over the following hours |
| `--backfill-ready` | off | walk every store once to queue pages found before the queue existed |
| `--domains` | none | a JSON list of the domains whose pages are queued, as names or `{host, weight, send_cap}`, read again when it changes; others are crawled but not queued |
| `--join-per-sec` | 100000 | stored URL records read per second while queuing the pages of domains new to `--domains`; 0 turns it off |
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
- **retries**: pages without a lastmod due for a re-crawl, and failed or dropped pages going again.

Each lane gets its share of what the domain sends (`--new-share`, `--retry-share`, and the rest for
changed pages), kept even across passes, and what one lane leaves unused goes to the others.

Every 5 seconds the bot asks `GET /v1/room` how many tasks the API can take, and sends that many
times 1,000 URLs, a few from each domain in turn, best ranked first, within `--per-domain-hourly` and
`--urls-per-hour`. The hourly pace starts empty and saves at most five minutes of sends, so a restart
or a stall is followed by no burst. A batch carries a hash of its content, so a batch sent again after
a timeout is queued once, and a batch in flight survives a restart. URLs over 2,048 characters, the
sitemap protocol's limit, are never sent.

A sent page with a lastmod goes again when its sitemap lists a newer one. A page without one goes
again `--recrawl-days` after it was sent, or after its domain's own interval. The outcome feed
`outcomes/seq/<n>.json` is read from its newest file on, up to 8 files at a time, and skips to the
newest when it falls an hour behind. Each file adds its published and failed pages to their domain's
yield, and the dispatcher gives domains whose pages rarely come back fewer slots, down to an hourly
probe. A failed page goes again after its domain's interval, and after two failures in a row waits
`--gone-days`; a dropped page, whose task never reached the site, goes again within the hourly budget.
A number still missing after two minutes, or whose file expired, is skipped. A status line every minute shows the queue, what was
sent in each lane, the room and the outcome feed.

With `--domains`, each listed domain may carry a `weight` (1 to 4) and a `send_cap`. A domain's turn
is worth its rank's slots times its weight. The bot counts the distinct pages it has sent for each
domain; at its `send_cap` the domain sends no new pages, and only changed pages and re-crawls go on.
A walker queues the pages each listed domain stored before it joined the list, newest first, as
many as its cap leaves, and walks a domain again when its cap changes. It reads `--join-per-sec`
records a second. A domain already sending is only counted, since its pages are queued already.

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
