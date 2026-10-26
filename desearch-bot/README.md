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
python tools/crawl_list.py --out domains.json --recrawl-out recrawl.txt   # --domains and --recrawl-overrides, from the registry
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
| `--new-share` | 50 | for a domain whose queue is not built yet, percent of its sends kept for pages never sent |
| `--retry-share` | 10 | for such a domain, percent kept for retries and re-crawls; changed pages get the rest |
| `--recrawl-days` | 7 | for a domain whose queue is not built yet, days after a send that a page without a lastmod, or a page that failed, goes out again |
| `--recrawl-overrides` | none | a file of `domain days` lines, such as `bbc.co.uk 1`, giving listed domains and their subdomains their own interval |
| `--gone-days` | 30 | days a page that failed on two sends in a row waits before it goes out again, unless its lastmod moves |
| `--dropped-per-hour` | 200000 | pages of lost or taken-back tasks sent again each hour; more are spread over the following hours |
| `--backfill-ready` | off | walk every store once to queue pages found before the queue existed |
| `--domains` | none | a JSON list of the domains whose pages are queued, as names or `{host, weight, send_cap, hourly, backlog}`, read again when it changes; others are crawled but not queued |
| `--join-per-sec` | 100000 | stored URL records read per second while building listed domains' due queues, eight domains at once; 0 turns it off |
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

With `--domains`, each listed domain has a set: the pages it may hold, up to its `send_cap`, or for
news its `backlog`. Every page in a set has exactly one due time, kept in the domain's due queue, and
every pass sends the most valuable due pages its room allows. First crawls, changed pages, retries and
refreshes are all due pages; there are no fixed shares between them.

- **Which pages:** pages a sitemap lists, newest first, never listing, search or file pages or pages
  robots disallows. A news site's home and section pages stay out.
- **Joining:** a new page joins while its domain's set has room, due at once. A news set rolls: a new
  article joins a full set and the page due last leaves. Other sets wait for room or a higher cap.
- **Due again:** a sent page falls due after its interval, by site type: news articles 12 hours
  while under two days old, then a day, a week and a month as they age; reference, docs, academic and
  public 7 days; media and community 14; company, organization and personal 30. A domain's line in
  `--recrawl-overrides` replaces its type's interval. A page whose sitemap lastmod moves past its last
  send falls due at once; one that moves to a date before its last send falls due a day after that
  send. A sitemap read that re-dates more than 2% of its pages, and more than 50, moves only its 10
  newest.
- **Failures:** a failed page falls due a day later; the second failure in a row takes it out of the
  set. A dropped page, whose task never reached the site, falls due within the hourly budget of
  `--dropped-per-hour`. A page the index does not take leaves the set for good.

Every 5 seconds the bot asks `GET /v1/room` how many tasks the API can take, and sends that many times
1,000 URLs within `--urls-per-hour`. A pass offers its room to due sites 5,000 at a time, most
valuable first: rank and the list's weight, raised by how overdue the queue front is for its kind, and
a front a whole interval overdue goes ahead of any value. Each site offers what is due within what its
hour allows, a site with nothing left of its hour waits, and a pass reads only the front of each
queue. Once every due site had its turn, room left pulls refreshes forward: the pages due soonest of
sites with a full yield share, never a page sent within the last day. `hourly` (or `--per-domain-hourly`) caps every domain's
sends in any hour, at most two minutes' worth at once (100 pages at least), so a site sees a steady
stream. The hourly pace starts empty and saves at most five minutes of sends, so a restart or a stall
is followed by no burst. A batch carries a hash of its content, so a batch sent again after a timeout
is queued once, and a batch in flight survives a restart. URLs over 2,048 characters, the sitemap
protocol's limit, are never sent.

The outcome feed `outcomes/seq/<n>.json` is read from its newest file on, up to 8 files at a time, and
skips to the newest when it falls an hour behind. Each file adds its published and failed pages to
their domain's yield, counted twice: fading over a day and over an hour. A domain whose pages rarely
come back gets fewer turns, down to 20 pages an hour, by the lower of the two, so a domain that starts
refusing slows within the hour. A number still missing after two minutes, or whose file expired, is
skipped. A status line every minute shows what went in each kind, the queues and how many had pages
due, the room and the outcome feed.

A walker builds each listed domain's queue once from its stored pages, paced at `--join-per-sec`
records a second across eight domains at a time, smallest domains first: every sent page that is not
gone, due as its schedule says or after its interval (an overdue one somewhere within its next
interval, so first crawls go first), and the newest pages never sent up to the cap, due at once; a set
with more sent pages than its cap keeps the newest. While a domain's queue is built its ready lists
stop sending, and its outcomes and sitemap changes wait for the queue and then go onto it; its ready
lists are dropped once the queue holds the domain. The walker admits more when a domain's cap rises,
sent pages a cut left out coming back on their refresh schedule, or when gone pages freed real room,
shrinks a set whose cap was cut, counts a set's size again from its queue whenever it walks the
domain, and stops mid-domain when the bot stops. Until its queue is built, a domain sends from its
ready lists: pages a sitemap listed first or with a newer lastmod, failed or dropped pages going
again, and pages without a lastmod on their re-crawl interval, by `--new-share` and `--retry-share`.

To go back to a release without due queues, stop the bot and put every queue's schedules back on the
waiting and ready lists before starting the earlier release; its walker then walks every listed
domain again and queues the pages it never sent:

```bash
sudo systemctl stop desearch-bot
desearch-bot unbuild-queues --buckets-dir /var/lib/desearch-bot/buckets
sudo systemctl start desearch-bot
```

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

A day of the scheduler runs on copies of real stores: `copy-buckets` copies chosen buckets key by key
from the running bot's, read-only beside it, and `replay` runs passes on a simulated clock against a
task API in the process while the list walker builds their queues at `--join-per-sec`. Outcomes are
drawn from a real day's per host and hour, each listed domain's sitemap is read about once a day with
a few pages re-dated and a few new, one site re-dates its whole sitemap every hour, and the due window
is cut to the share of buckets copied. It reports each hour's sends by kind, pass times, memory, the
walker's progress and read rate, any site over its hourly limit or its two-minute spread, and any page
sent again within a day that no failure, drop or sitemap change explains.

```bash
desearch-bot copy-buckets --from /mnt/desearch-bot/buckets --to /mnt/replay/buckets --buckets 1,5,41
desearch-bot replay --buckets-dir /mnt/replay/buckets --buckets 1,5,41 --domains domains.json \
  --outcomes day-rates.tsv --start "$(date +%s)" --per-hour 234375 --log replay.txt
```
