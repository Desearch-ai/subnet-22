# desearch-bot

The DesearchBot crawl loop. For every domain it owns, it reads robots.txt and the sitemaps on a
schedule, keeps every URL they list, and reports each visit to the registry. One process uses every
core.

- **Stores**: 256 RocksDB bucket stores holding domains, sitemaps and URLs. The task API's feeder
  reads new URLs from them.
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
| `--duration` | none | stop after this many seconds |

`desearch-bot run --help` lists the memory and disk limits.

| Variable | |
| --- | --- |
| `DESEARCH_DB` | the registry's Postgres DSN |
| `DESEARCH_SIGNING_KEY_FILE` | the Ed25519 key requests are signed with |
| `MALLOC_ARENA_MAX=2` | stops glibc from holding on to memory RocksDB has freed |

SIGTERM and SIGINT stop the loop; visits still running after 30 seconds are dropped and simply fall
due again.

## Tests

```bash
cargo test --release
```

The parity tests replay fixed vectors for URL keys, joins, dates, signatures and language, byte for
byte. `tests/registry.rs` needs `initdb` and `postgres` on the PATH, and `tests/world.rs` crawls a
small local web end to end.
