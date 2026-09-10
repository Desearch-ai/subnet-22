# desearch-bot-rs

The DesearchBot crawl loop in Rust. For every domain in the buckets it owns, it reads robots.txt and the sitemaps, keeps every URL they list, and tells the registry what it found. It does the same work as `python -m desearch_bot.cli run`, with one process using every core.

It shares everything with the Python tools, so either crawler can run on the other's data:

- **Bucket stores**: the 256 RocksDB stores under `/var/lib/desearch-bot/buckets`, with the same keys and records (`D` domains, `S` sitemaps, `U` URLs, `M` sync mark).
- **Registry**: `bot.domains` in Postgres. Visits are reported in bulk every 30 seconds, and changes come back keyed on `(changed_at, host)`.
- **Rules**: robots.txt and Crawl-delay, one request a second per host, sitemap schedules and date trust, domain states, exclusions, and Web Bot Auth signatures.

Candidates, categories, Radar and the Hugging Face export stay in the Python package.

## Build

```bash
cargo build --release
```

RocksDB is compiled from source, which needs `clang` and a C++ toolchain.

## Run

```bash
desearch-bot run --concurrency 840 --buckets-dir /var/lib/desearch-bot/buckets --data-dir data
```

| Option | Default | Meaning |
| --- | --- | --- |
| `--concurrency` | 840 | Visits in flight at once |
| `--buckets` | `0-255` | Buckets to crawl, as ranges such as `0-13,20` |
| `--buckets-dir` | `/var/lib/desearch-bot/buckets` | Where the bucket stores live |
| `--data-dir` | `data` | Where `public_suffix_list.dat` lives |
| `--cache-mb`, `--memtable-mb` | 4096, 2048 | Block cache and memtable budget shared by all stores |
| `--timeout` | 10 | Seconds a read may stall before the request fails |
| `--no-registry` | off | Crawl without reporting to Postgres or taking changes from it |
| `--duration` | none | Stop after this many seconds |

Environment:

- `DESEARCH_DB`: the registry DSN.
- `DESEARCH_SIGNING_KEY_FILE`, or `DESEARCH_SIGNING_KEY` inline: the Ed25519 key that signs requests.

DNS goes to the local caching resolver on `127.0.0.1:5335`, and answers that point into private networks are dropped. SIGTERM and SIGINT stop the loop. Visits still running after 30 seconds are dropped, and those domains simply fall due again.

## Tests

```bash
cargo test --release
```

The parity tests replay vectors that the Python code produced, and must match it byte for byte:

- URL keys for real sitemap URLs
- URL joins
- `lastmod` parsing
- request signatures
- suffix groups and exclusion reasons

Set `PSL_FILE` to a copy of the public suffix list to check registrable domains too. `tests/world.rs` crawls a small local web end to end: robots.txt, a sitemap index, a gzipped sitemap, conditional requests, a blocked site, a site with no sitemap, and a site whose DNS fails.

`examples/digest.rs` times one sitemap file through decompressing, parsing, dating, normalising and storing, the CPU work of a visit:

```bash
cargo run --release --example digest -- sitemap.xml example.com /tmp/digest-store
```
