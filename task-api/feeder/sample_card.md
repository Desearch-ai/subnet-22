---
pretty_name: SN22 sandbox URLs
size_categories:
- 1M<n<10M
configs:
- config_name: default
  data_files: data/*.parquet
---

# SN22 sandbox URLs

URLs for testing a [Desearch Subnet 22](https://github.com/Desearch-ai/subnet-22) miner locally,
before registering on mainnet.

Run `python -m sandbox` in the subnet-22 repository. It downloads these files, queues the URLs
in a local task API as tasks, and runs a local validator that checks your miner's
uploads the way mainnet validators do. Point your miner at the local API and watch each verdict,
the pages that did not match and why, and your throughput. The
[guide](https://github.com/Desearch-ai/subnet-22/blob/desearch-2.0/docs/test-your-miner.md)
has the steps.

Each row is one page URL, in `data/urls-NNNNN.parquet` files of 100,000 rows. The rows are
shuffled, so every file mixes many sites and one file is enough for a test.
