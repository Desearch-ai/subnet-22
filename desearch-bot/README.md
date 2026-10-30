# desearch-bot

The DesearchBot crawl loop. For every domain it owns, it reads robots.txt, the sitemaps and the feeds
on a schedule, keeps every URL they list, and reports each visit to the registry. It also sends new and
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
python tools/crawl_list.py --out domains.json   # --domains, from the registry
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
| `--heavy-slots` | 16 | visits at once to domains with more than 2,000 sitemap records |
| `--task-api` | none | the task API to fill, such as `https://api-22.desearch.ai` |
| `--outcomes-url` | none | where the task API publishes what became of each URL, such as `https://subnet-22.s3.us-east-va.io.cloud.ovh.us` |
| `--urls-per-hour` | 0 | URLs sent in an hour at most, at a steady pace; 0 sends what the task API has room for |
| `--per-domain-hourly` | 2000 | URLs one domain may send in an hour |
| `--new-share` | 50 | percent of each domain's sends kept for pages never sent |
| `--retry-share` | 10 | percent kept for retries and scheduled re-crawls; changed pages get the rest |
| `--recrawl-days` | 7 | days after a send that a page without a lastmod, or a page that failed, goes out again |
| `--recrawl-overrides` | none | a file of `domain days` lines, such as `bbc.co.uk 1`, giving listed domains and their subdomains their own interval |
| `--gone-days` | 30 | days a page that failed on two sends in a row waits before it goes out again, unless its lastmod moves |
| `--dropped-per-hour` | 200000 | pages of lost or taken-back tasks sent again each hour; more are spread over the following hours |
| `--backfill-ready` | off | walk every store once to queue pages found before the queue existed |
| `--domains` | none | a JSON list of the domains whose pages are queued, as names or `{host, type, weight, send_cap, hourly, backlog, since, resend_before, depth}`, read again when it changes; others are crawled but not queued |
| `--join-per-sec` | 100000 | stored URL records read per second while queuing the pages of listed domains; 0 turns it off |
| `--resend-per-sec` | 20000 | pages listed per second to go once more for domains whose `resend_before` changed; 0 turns it off |
| `--inbox` | none | a directory of URL lists, one `<host>.txt` per host |
| `--backfill-sent` | none | pages the Python feeder sent, as tab-separated host, path, lastmod and time, marked sent first |
| `--duration` | none | stop after this many seconds |

`desearch-bot run --help` lists the memory and disk limits.

| Variable | |
| --- | --- |
| `DESEARCH_DB` | the registry's Postgres DSN |
| `DESEARCH_SIGNING_KEY_FILE` | the Ed25519 key requests are signed with |
| `FEEDER_KEY_URI` | the task API admin hotkey, as a Substrate secret URI, needed with `--task-api` |
| `MALLOC_ARENA_MAX=2` | stops glibc from holding on to memory RocksDB has freed |

## Visits

Visits share their slots between re-checks of known sites, sitemap backlogs and domains never
checked. Each kind can count on its share, and slots one leaves unused go to the others. Due domains
go listed news sites first, then the rest of the list, then unlisted domains, by rank within each.
Domains with more than 2,000 sitemap records take one of `--heavy-slots`, handed out from a waiting
queue in the same order, where each hour of waiting moves a site up a class. A registry row set back
to `new` has its site checked again.

A sitemap read again looks up only the entries it did not list the same way last time, from a compact
fingerprint set kept per sitemap. A listed news site's root sitemap index and its Google News sitemaps
are read again within 30 minutes.

RSS and Atom feeds are read as sitemaps. Where robots.txt names no sitemap, the bot tries
`/sitemap.xml` and `/sitemap_index.xml`, then on a listed site `/feed`, `/rss.xml` and its homepage;
an answer that is neither a sitemap nor a feed isn't kept. A homepage is read as a list of itself and
the pages on its site it links to with no query, from its first 400 links and its first 512 KB,
relative links taken from where it answered; it counts as changed only when those pages change, and
is read again at most hourly. It never stands in for a sitemap: the guesses are tried again on later
visits. A site with a `depth` has its homepage read this way beside whatever sitemaps it has.

A listed subdomain with a registry record of its own is crawled on its own host, under its own
robots.txt. Its pages go under it while that record is active or failing, and not at all while it is
blocked, excluded or ineligible. In any other state, and while its homepage is missing, they stay
with its parent. The bot reads which subdomains stand on their own before its first send and every 10 minutes.

With `--inbox DIR`, each `<host>.txt` in the directory, a URL a line with optionally a tab and its
date, is taken in as that host's listing, in parts of 50,000 URLs, the way a sitemap's pages come in.
The bot looks every 5 minutes and takes only files changed since it last took them; each file's time
is kept in the store, so a restart reads nothing again. A file for a host without a registry record
waits for the record, and a file that can't be read is skipped.

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
`--urls-per-hour`. A domain sends at most two minutes' worth of its hourly limit at once (100 pages at
least), so a site sees a steady stream rather than its hour in one burst. A site whose robots.txt
names a Crawl-delay is held to one page per delay over the hour, the delay counted up to a minute, and
sends them one at a time. The hourly pace starts empty and saves at most five minutes of sends, so a
restart or a stall is followed by no burst. A batch carries a hash of its content, so a batch sent
again after a timeout is queued once, and a batch in flight survives a restart. URLs over 2,048
characters, the sitemap protocol's limit, are never sent.

A sent page with a lastmod goes again when its sitemap lists a newer one. A page without one goes
again `--recrawl-days` after it was sent, or after its domain's interval in `--recrawl-overrides`,
which the bot reads when it starts. The outcome feed `outcomes/seq/<n>.json` is read from its newest
file on, up to 8 files at a time, and skips to the newest when it falls an hour behind. Each file adds
its published and failed pages to their domain's yield, counted twice: fading over a day and over an
hour. The dispatcher gives domains whose pages rarely come back fewer slots, down to an hourly probe,
by the lower of the two, so a domain that starts refusing slows within hours. A failed page goes again
after its domain's interval, and after two failures in a row waits `--gone-days`; a dropped page,
whose task never reached the site, goes again within the hourly budget. A number still missing after
two minutes, or whose file expired, is skipped. A status line every minute shows the queue, what was
sent in each lane, the room and the outcome feed.

Room the lists leave in a pass goes to refreshes: pages already sent, of listed domains with nothing
waiting whose pages come back (at least 200 outcomes judged, and no slowing for yield), oldest send
first. A background keeper, paced like the walker, keeps a stock of each such domain's pages sent
longest ago (a quarter of its hourly limit, 1,000 at least) and chooses it again when a pass finds it
empty, reading at most 200,000 sent records a choice from where the last one stopped; a pass reads
only the stock, never the sent records. A page sent within the last day stays, and a stocked page
that went out again or is now failing, requeued or changed leaves the stock unsent. A pass offers
room to 2,000 domains in turn, and refreshes stay within each domain's hourly limit and its
two-minute spread.

## The crawl list

With `--domains`, each listed domain may carry a `type`, a `weight` (1 to 4), a `send_cap`, an
`hourly` limit, a `backlog`, a `since` date, a `resend_before` date and a `depth`. A domain's turn is worth its
rank's slots times its weight, and `hourly` replaces `--per-domain-hourly` for it. A domain typed
`news` gets the news visit order and re-reads above.

**Caps.** The bot counts the distinct pages it has sent for each domain; at its `send_cap` the domain
sends no new pages, and only changed pages and re-crawls go on. A walker queues the pages each listed
domain stored before it joined the list, newest first, as many as its cap leaves or, for a domain
without a cap, up to its `backlog`, smallest domains first and the higher weight first among domains
of a size. It reads `--join-per-sec` records a second and stops mid-domain when the bot stops. A
domain already sending is counted when its cap falls, so the cap applies at once, and walked again
when its cap rises or its floor falls, keeping the pages found while the walk ran.

**Date floors.** `since` is the oldest page date a domain sends. A page's date is its lastmod, with a
day's grace for time zones, or else a date in its path (`/2025/03/14/`, `/2025/03/`, `2025-03-14` or
`20250314`); an undated page goes. Pages dated before the floor are left unqueued at sitemap reads,
in the walk and in the backfill, and stay out of the refresh stock; pages already waiting are dropped
as they would go, which the status line counts. A floor that only rises needs no walk.

**Depth.** A landing site's `depth` is the most path levels its pages may have, a leading English
locale (`/en/`, `/en-us/`) not counted, as the v1 set's landing rule reads them. Deeper pages, pages
with a query, pages under another locale (`/de-de/`) and pages of other hosts stay unsent, wherever
the date floor leaves pages out, and its cap counts only the pages its depth lets go. A deeper depth,
or none, walks the domain again.

**Resends.** `resend_before` sends a domain's pages last sent before it once more, as after an
extraction change. When a domain's value changes, a walk in a thread of its own lists those pages at
`--resend-per-sec` a second, newest lastmod first, on a resend list apart from the lanes; a list
built again with the same value walks nothing. Failing pages, pages robots.txt disallows and pages
dated before the floor stay off. A site with a `depth` goes again only up to its `send_cap`, its
shallowest pages first, so its homepage leads. Resends take only the room the lanes leave, before refreshes, within
the domain's hourly limit and Crawl-delay, and refreshes pass a domain by while its resends wait. A
failed resend gets no retry of its own. Resent pages' outcomes stay out of the yields that pace the
domain, also across a restart: the pages lately resent are kept in `resent-pages.bin` beside the
stores. With the bot stopped, `desearch-bot unresend` takes every waiting resend back, clears the
pages' marks and forgets each domain's walked value; install a list without `resend_before` first.

`tools/crawl_list.py` builds the list from the registry. `tools/v1_set.py` writes the v1 set, each
v1 site with its rule; `crawl_list.py --v1-set` lists its sites first under their caps and floors.
`tools/resend_waves.py` splits the sites with pages to send again into waves, and `crawl_list.py
--resend-before DATE --resend-hosts FILE` gives the hosts of the waves applied so far their
`resend_before`. `tools/build-v1.sh` runs the three on the bot box.

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

The dispatcher also runs on copies of real stores. `copy-buckets` copies chosen buckets key by key
from the running bot's, read-only beside it, and `replay` runs passes on a simulated clock against a
task API in the process, with the list walker, the resend walk and the stock keeper running beside
them. Outcomes are drawn from a real day's per host and hour, each listed domain's sitemap is read
about once a day with a few pages re-dated and a few new, and one site re-dates its whole sitemap
every hour. `--switch-at` and `--switch-domains` change the crawl list mid-run. Each hour it reports
sends by kind, pass times, the walks' progress, memory, and any site over its hourly limit or
two-minute spread.

```bash
desearch-bot copy-buckets --from /var/lib/desearch-bot/buckets --to /mnt/replay/buckets --buckets 1,5,41
desearch-bot replay --buckets-dir /mnt/replay/buckets --buckets 1,5,41 --domains domains.json \
  --outcomes day-rates.tsv --start "$(date +%s)" --hours 8 --per-hour 234375 --log replay.txt
```
