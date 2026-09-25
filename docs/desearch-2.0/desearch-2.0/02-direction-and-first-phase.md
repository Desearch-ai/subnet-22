# Desearch 2.0: direction and first phase

We plan to turn network contributions into a maintained web collection, a searchable index and better retrieval technology. The [1.0 story](01-why-desearch-2.0.md) explains why that change matters.

## What we intend to build

Our source scope extends across company websites, news, articles, blogs, documentation and other useful public-web pages. We will start with selected domains and workloads, not attempt to index the entire web in the first phase.

The intended system connects five kinds of work:

| Work                                            | Why it matters                                                                    |
| ----------------------------------------------- | --------------------------------------------------------------------------------- |
| Collect and refresh useful pages                | Build coverage and keep changing information current                              |
| Clean, filter, deduplicate and organize content | Reduce repeated or low-value material and retain information applications can use |
| Maintain source records and indexes             | Connect content to its origin and version, and make it retrievable                |
| Improve embeddings and ranking                  | Help queries find relevant information and put the most useful results first      |
| Serve tested versions                           | Deliver dependable search while new work continues in the background              |

The underlying collection should retain useful source information independently of a particular embedding model. An embedding is a numerical representation used for matching meaning; it is not a replacement for the source text.

## Where we are now

This is the proposed 2.0 direction, not a report of a deployed system, completed testnet or open competition. The history describes 1.0; this chapter describes what comes next.

No new reward program or operator migration is opened by this proposal. Actual implementation and test results will be reported as they become available.

## The first planned work

**Crawl and refresh a bounded collection.** Contributors fetch permitted pages and extract content. The preparation process checks source records, filters unwanted material, removes duplicates and organizes accepted text.

**Compute embeddings with a selected model.** Contributors run a specified model version on approved text to prepare it for retrieval. This is execution work, not training or inventing a model.

**Build and test the searchable foundation.** Accepted content and representations enter a maintained index. We compare the resulting search against a reproducible baseline before routing customer traffic to it.

Initially, the team would operate the work queue, stored collection, index and customer-serving system. Miners would build and maintain the inputs to the index outside the customer query path. This is not a promise of decentralized storage or live query serving by miners.

An ordinary indexed query should use a tested version without waiting for a miner to complete a new task or a validator to make a decision. Updates enter through assessment and release review, with monitoring and rollback. The speed and quality benefit still needs measurement.

The [first-phase technical overview](03-first-phase-technical-overview.md) explains task inputs and outputs, validator checks and the path into customer search. Operators should also read the [GPU planning notice](05-participation-and-progress.md#hardware-planning-for-the-first-phase).

## The first reviewable checkpoint

The first checkpoint should show:

1. A bounded source collection and representative queries, with a reason those sources and questions matter.
2. Inspectable samples of accepted content and source records, plus the baseline and evaluation specification.
3. An assessment of readiness for the early reranking experiment: suitable inputs and evaluation, or the concrete gap preventing it.

At that checkpoint, we would report useful results, retrieval-only response time, source coverage and freshness, data quality, and full build/check/serve cost. Retained data should be usable and maintained, not merely counted.

This is the evidence we propose to prepare, not a claim it exists already. Program resources and numerical thresholds must be selected before the relevant experiment.

## How the network's work can expand

CRAWL and EMBED are the first task families, not the final definition of mining on SN22. We plan to expand beyond them so contributors can improve both the data and the technology used to search it. Possible later programs include:

| Later contribution opportunity                 | Intended result                                                         |
| ---------------------------------------------- | ----------------------------------------------------------------------- |
| Train or fine-tune retrieval models            | Better models for matching questions to relevant content                |
| Develop ranking and reranking methods          | Better ordering of candidate results under useful speed and cost limits |
| Improve extraction, filtering and organization | More useful text, categories and source-linked information              |
| Improve coverage and refresh methods           | Find missing sources and keep important information current             |

The expansion is our intended direction; specific programs, their sequence and terms are not yet set. The list is not exhaustive: further tasks can emerge from customer needs and useful contributor proposals. Each new program will have its own inputs, requirements, evaluation and reward terms announced before work opens. The [incentives overview](04-incentives-and-rewards.md#how-incentives-can-expand-after-the-first-phase) explains why execution tasks and method competitions need different assessment.

Model builders could train or fine-tune appropriately licensed open-source models and submit reproducible candidates. Candidates must work with the index. Improvement claims are tested against a competent baseline and the declared parent model where applicable. Adoption depends on useful gains within the program's quality, speed and cost limits; a strong offline score alone does not authorize deployment.

A bounded reranking experiment within SN22 remains the early candidate for testing method improvement. Its timing and terms are not set, and no contest is opened here. Preparation can proceed alongside the initial build when suitable inputs and an evaluator are available.

## What the same foundation can support

Fast search is the first product path. Later, the organized source collection could support structured datasets about companies, maintained information feeds or suitable data for training and evaluating other AI systems, including other subnets.

Those are applications of the same foundation, not a commitment to launch them all at once. Each needs appropriate preparation, customer demand and usage rights. Additional storage or serving roles may also be explored where they improve outcomes; no particular topology is a required destination.

The [network rationale](../company/open-network-and-bittensor.md) explains why Bittensor and how contributions can be adopted. Continue to the [first-phase technical overview](03-first-phase-technical-overview.md) for task inputs, validator checks and the path into search.
