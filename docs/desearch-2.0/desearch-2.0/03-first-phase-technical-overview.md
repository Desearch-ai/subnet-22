# Desearch 2.0: first-phase technical overview

This chapter explains the first-phase design for miners, validators and technical readers. Crawling and embedding run as described in [how it works](../../architecture.md), [embedding tasks](../../embedding-tasks.md) and the [miner](../../miner.md) and [validator](../../validator.md) guides. [Direction and first phase](02-direction-and-first-phase.md) explains the scope and what comes later.

## Two paths: building the collection and answering queries

The key change is to prepare reusable information before a customer needs it. A contribution can improve the collection used by many searches; its author does not need to serve each request.

```text
Build and refresh
  Team selects sources and issues tasks
    → miners crawl pages and submit source-linked content
    → validators assess crawling outputs
    → content preparation produces approved text chunks
    → miners compute embeddings with the selected model
    → validators assess embedding outputs
    → team builds and tests a candidate index version
    → approved version becomes available to customer search

Customer search
  Query → active, tested index and models → ranked sources and supporting text
```

Validators assess both crawling and embedding work; the checks at each step are described below. The team operates the queue, stored collection, index and API. Ordinary indexed queries use an approved version without waiting for new miner work or a validator decision, so preparing the collection stays out of the query path.

## What a miner receives and returns

The initial task families are `CRAWL` and `EMBED`. A task identifies its inputs, required configuration, expected outputs and applicable assessment criteria. Miners retrieve assigned work and submit outputs with their task and contributor identity.

| Task    | Input                                                                                                | Expected output                                                                                                 |
| ------- | ---------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------- |
| `CRAWL` | Selected URLs, permitted-use instructions, relevant prior source information and extraction settings | Fetch status, permitted page content and extracted text, source URL and observation time    |
| `EMBED` | Approved text chunks and their identifiers, a selected model version and representation settings     | Numerical vectors linked to the input chunks in the required order, with the model and configuration identified |

A chunk is a piece of source text prepared for retrieval. An embedding represents that text numerically so search can match meaning, not only exact words. First-phase embedding miners run the selected model; they do not train a replacement or submit vectors from arbitrary models.

The [hardware notes](05-participation-and-progress.md#hardware) set out GPU needs for embedding miners and validators checking embeddings.

## What validators check

For **crawling**, validators re-extract a sample of the uploaded pages from their HTML, fetch the same pages themselves and compare the text, and check a sample of the pages the miner reported as failed. A page the validator cannot load either is not held against the miner.

For **embeddings**, the task API refuses a file without exactly one vector per text, and validators embed a sample of the texts again with the same model. Each vector must have finite values and length 1 and come close enough to theirs by cosine similarity, which allows for numerical differences across GPUs rather than requiring identical bytes.

The exact sampling, tolerances and handling of pages that cannot be loaded are in [emission](../../emission.md) and [embedding tasks](../../embedding-tasks.md#how-it-is-checked). A sampled check does not establish that every item was independently verified.

Task assessment and customer deployment remain separate. [Incentives and rewards](04-incentives-and-rewards.md#from-completed-work-to-rewards) explains the path from checked work to weights and rewards; the team separately approves versions for customer traffic.

## How accepted work becomes searchable

The team oversees content preparation: filtering low-value content, removing duplicates and splitting accepted text into page texts and passages. The collection keeps source text, URLs, fetch times and version references independently of embeddings, so replacing a model does not lose the information used to create its vectors.

Search combines keyword retrieval for exact terms with vector retrieval for meaning. Ranking combines and orders candidate results. A search version binds the collection and text preparation to compatible document/query models, indexes and ranking settings. Vectors from unrelated models cannot simply be mixed because their sizes match.

Accepted work enters a candidate version for evaluation, not directly into live search. Before promotion, the team tests useful results, response time, data quality and operating cost. Refreshes and model changes must preserve source/version links and support rollback without restoring deleted material.

## Operator guides

Installation, configuration, task formats, the model, hardware and the checks are in the [miner](../../miner.md) and [validator](../../validator.md) guides, [embedding tasks](../../embedding-tasks.md) and [emission](../../emission.md).

Training, fine-tuning and competitions for better retrieval methods belong to the [wider direction](02-direction-and-first-phase.md#how-the-networks-work-can-expand). They are not part of the CRAWL and EMBED tasks.

Continue to [incentives and rewards](04-incentives-and-rewards.md) for how assessed work counts toward rewards.
