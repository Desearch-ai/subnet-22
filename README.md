<h3 align="center">
  <a name="readme-top"></a>
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="./docs/assets/desearch-logo.png">
    <img src="./docs/assets/desearch-logo-black.png" alt="Desearch" width="360">
  </picture>
</h3>

<div align="center">
  <a href="./LICENSE"><img src="https://img.shields.io/github/license/Desearch-ai/subnet-22" alt="License"></a>
  <a href="https://github.com/Desearch-ai/subnet-22/graphs/contributors"><img src="https://img.shields.io/github/contributors/Desearch-ai/subnet-22.svg" alt="Contributors"></a>
  <a href="https://www.desearch.ai/network"><img src="https://img.shields.io/badge/Bittensor-Subnet%2022-blue" alt="Bittensor Subnet 22"></a>
  <a href="https://desearch.ai"><img src="https://img.shields.io/badge/Visit-desearch.ai-orange" alt="Visit desearch.ai"></a>
</div>

<p align="center">
  <a href="https://x.com/desearch_ai"><img src="https://img.shields.io/badge/Follow%20on%20X-000000?style=for-the-badge&logo=x&logoColor=white" alt="Follow on X"></a>
  <a href="https://www.linkedin.com/company/desearch-ai/"><img src="https://img.shields.io/badge/Follow%20on%20LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white" alt="Follow on LinkedIn"></a>
  <a href="https://discord.com/invite/eb6DTZNMF5"><img src="https://img.shields.io/badge/Join%20our%20Discord-5865F2?style=for-the-badge&logo=discord&logoColor=white" alt="Join our Discord"></a>
</p>

---

# Desearch

**A web index for AI, built through open competition on Bittensor.**

AI applications need relevant, current sources and text they can use, soon enough to finish their
task. Desearch collects and refreshes useful pages from the public web, prepares them for retrieval
and develops the models that help AI find the right sources. Search is the first application: fast
search for AI agents, and deeper research for questions that take longer.

The work happens in the open on Bittensor Subnet 22. Miners build and refresh the collection,
validators check their work under published rules, and the Desearch team integrates accepted work
into a tested index and the search API.

<p align="center">
  <a href="./docs/miner-setup.md">⛏️ Mine</a> ·
  <a href="./docs/validator-setup.md">🛡️ Validate</a> ·
  <a href="./docs/architecture.md">⚙️ How it works</a> ·
  <a href="./docs/desearch-2.0/README.md">🧭 Desearch 2.0</a> ·
  <a href="https://console.desearch.ai">🔑 Search API</a>
</p>

---

## Why an index

In Desearch 1.0, miners answered search requests and validators judged the answers. It worked, but
good answers could arrive too late, some depended on outside search providers, and little of the
work left anything that could be kept and improved.

Desearch 2.0 prepares information before a query arrives. Pages are collected, cleaned,
deduplicated, organized and prepared for retrieval in the background, so a query searches a
maintained collection instead of starting from scratch. That gives:

- **Faster retrieval**: expensive preparation moves out of the query path
- **More control over quality**: coverage, freshness and supporting text can be inspected, and models
  improved against the same collection
- **Work that lasts**: every accepted page serves many later searches, not one answer

The collection covers the public web: company websites, news, articles, blogs and documentation. It
starts with selected sources and grows as results prove useful.

## What the network builds

| Work | Who does it | Status |
| --- | --- | --- |
| **Crawl** and refresh pages, extract their text | Miners fetch assigned pages through their own proxies; validators re-fetch a sample of every upload | Running on SN22 |
| **Embed** accepted text with a selected model | Miners run the model on their GPUs; validators recompute a sample | Built; opens with Desearch's embedding model |
| Train retrieval models, improve ranking, extraction and coverage | Model builders and data operators | Later programs, announced before they open |
| Build, test and serve the index | The Desearch team | Team-operated; queries never wait on miner work |

## Mining and validating

**Miners** crawl: they claim tasks, fetch the assigned pages through their own proxies and upload the
extracted text. They are paid by their share of verified pages. [Miner setup →](./docs/miner-setup.md)
· [Emission →](./docs/emission.md)

**Validators** check: every validator re-fetches a sample of every upload, reports pass or fail, and
sets weights from what its own checks found. [Validator setup →](./docs/validator-setup.md)

Crawl rounds are committed to a future block before tasks go out, every step is logged and signed,
an upload is paid and published only when a majority of validators agree, and every result is
[public](https://task-api.desearch.ai/v1/tasks).
[Architecture →](./docs/architecture.md)

## Use Desearch

The search API is available today for developers building AI agents and research tools. Get a key in
the [console](https://console.desearch.ai) and start with the
[documentation](https://www.desearch.ai/docs/guide/introduction/desearch-ai).

| SDK | Install |
| --- | --- |
| [Python](https://github.com/Desearch-ai/desearch.py) | `pip install desearch-py` |
| [JavaScript / TypeScript](https://github.com/Desearch-ai/desearch.js) | `npm install desearch-js` |
| [MCP server](https://github.com/Desearch-ai/mcp-desearch) | `npm install -g desearch-mcp-server` |

## Documentation

| Guide | What's inside |
| --- | --- |
| [Miner setup](./docs/miner-setup.md) | Install, register, configure, run, how you earn, monitoring |
| [Validator setup](./docs/validator-setup.md) | Install, register, configure, run, automatic upgrades, monitoring |
| [Emission](./docs/emission.md) | How miners' shares are worked out and what raises them |
| [Architecture](./docs/architecture.md) | How the bot, task API, miners, validators, storage and engine work together |
| [Embedding tasks](./docs/embedding-tasks.md) | What embed tasks will contain and how they are checked, before they open |
| [Engine](./engine/README.md) | The search index and API, and how new pages reach it |
| [Task API](./task-api/README.md) | Endpoints, rounds, scoring rules, audits, public logs and the storage layout |
| [Desearch 2.0](./docs/desearch-2.0/README.md) | The direction: why an index, the first phase, incentives and participation |

## Contributors

<a href="https://github.com/Desearch-ai/subnet-22/graphs/contributors">
  <img alt="Contributors" src="https://contrib.rocks/image?repo=Desearch-ai/subnet-22">
</a>

## Community

- [Discord](https://discord.com/invite/eb6DTZNMF5): questions, mining and validating support
- [X](https://x.com/desearch_ai) and [Telegram](https://t.me/desearchAI): announcements
- [Blog](https://www.desearch.ai/blog): guides and engineering write-ups

## License

Released under the [MIT License](./LICENSE).

<p align="right"><a href="#readme-top">↑ Back to top</a></p>
