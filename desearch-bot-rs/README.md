# desearch-bot-rs

The DesearchBot crawl loop in Rust. For every domain it owns, it reads robots.txt and the sitemaps on
a schedule, keeps every URL they list, and reports each visit to the registry. It does the same work
as `python -m desearch_bot.cli run`, in one process that uses every core.

It shares its data and rules with the Python package, so either can run on the other's data:

- **Stores**: 256 RocksDB bucket stores holding domains, sitemaps and URLs.
- **Registry**: the `bot.domains` table in Postgres, which records each domain's state.
- **Rules**: robots.txt and Crawl-delay, one request a second per host, sitemap schedules, exclusions,
  and Web Bot Auth request signatures.

Finding new domains, categories and the Hugging Face export stay in the Python package.

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

The parity tests check the Rust code against output from the Python package, byte for byte.
`tests/registry.rs` needs `initdb` and `postgres` on the PATH, and `tests/world.rs` crawls a small
local web end to end.
