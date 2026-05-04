<div align="center">

<img src="./docs/assets/desearch-logo.png" alt="Desearch" width="480" />

# **Subnet 22 on Bittensor**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

</div>

Welcome to **Desearch powered by Bittensor Subnet 22**! Desearch is a decentralized,
AI-powered search engine that returns unbiased and verifiable results across X, Reddit,
Arxiv, Hacker News, Wikipedia, YouTube, and the broader web. Frontend and API access are
available at [desearch.ai](https://desearch.ai).

## Table of Contents

- [Introduction](#introduction)
- [Key Features](#key-features)
- [High-Level Architecture](#high-level-architecture)
- [Getting Started](#getting-started)
  - [For API Consumers](#for-api-consumers)
  - [For Miners](#for-miners)
  - [For Validators](#for-validators)
- [Monitoring](#monitoring)
- [Contact and Support](#contact-and-support)

## Introduction

Desearch delivers an unbiased, verifiable search experience built on the Bittensor
network. Miners compete to return the best search results from multiple sources;
validators independently verify result quality and assign on-chain rewards. Through
the public API, developers and AI builders integrate real-time, decentralized search
into their products.

## Key Features

- **AI-powered analysis** — decentralized models produce relevant, contextual, unfiltered results.
- **Diverse data sources** — X, Reddit, Arxiv, Hacker News, Wikipedia, YouTube, and general web.
- **Sentiment and metadata analysis** — captures emotional tone and key metadata for social content.
- **Verifiable rewards** — validators independently scrape and score miner outputs.
- **Extensible** — community-driven improvements to scoring, sources, and relevance.

## High-Level Architecture

- **Miners** run a single Bittensor **axon** that answers `IsAlive` and all search synapses
  (AI / Twitter / Web). Validators call the axon directly via dendrite.
- **Validators** generate synthetic queries every UTC hour, dispatch them to miners,
  independently verify results against ground-truth scrapers (Apify, ScrapingDog), and
  write weights on-chain. They also expose an organic-search FastAPI that the Desearch
  product backend calls on behalf of API consumers.
- **Bittensor network** — settles miner compensation on-chain in $TAO.

## Getting Started

### For API Consumers

To integrate Desearch into your product, visit [desearch.ai](https://desearch.ai) and
request an API key. Consumers send requests to the Desearch API with their API key; the
Desearch backend routes those requests to validators on your behalf and returns the
aggregated search results.

### For Miners

Miners contribute search capacity and are rewarded based on result quality and volume.
Expected setup steps:

- Prepare a server with Python ≥ 3.11, PM2, and a registered hotkey on netuid 22.
- Configure credentials for OpenAI, SerpAPI, and Apify.
- Declare per-search-type concurrency in `neurons/miners/manifest.json`.
- Run the axon with PM2.

See the [Miner Setup Guide](./docs/running_a_miner.md) for full instructions.

### For Validators

Validators verify miner outputs and write weights on-chain. Expected setup steps:

- Prepare a server with Python ≥ 3.11, PM2, Redis, `jq`, and a registered validator hotkey.
- Configure credentials for OpenAI, Apify, ScrapingDog, and W&B.
- Generate a public API access key and run the autoupdate script.

See the [Validator Setup Guide](./docs/running_a_validator.md) for full instructions.

### Additional Guides

- [Environment Variables](./docs/env_variables.md)
- [Testnet Operations](./docs/running_on_testnet.md)
- [Mainnet Operations](./docs/running_on_mainnet.md)

## Monitoring

Validators stream metrics to Weights & Biases. Public dashboards are available at
[wandb.ai/smart-scrape/smart-scrape-1.0](https://wandb.ai/smart-scrape/smart-scrape-1.0).

## Contact and Support

- **Website** — [desearch.ai](https://desearch.ai)
- **Subnet 22 channel** — [Bittensor Discord](https://discord.com/channels/799672011265015819/1189589759065067580)
- **Desearch Discord** — [Join the Desearch community server](https://discord.com/invite/eb6DTZNMF5)
