# Desearch 2.0: first-phase technical overview

This chapter explains the proposed first-phase design for miners, validators and technical readers. It is not an installation guide or a claim that this system is live. [Direction and first phase](02-direction-and-first-phase.md) explains the scope and what comes later.

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

Validators assess both crawling and embedding work; the checks at each step are described below. The team would initially operate the queue, stored collection, index and API. Ordinary indexed queries would use an approved version without waiting for new miner work or a validator decision. This separation is intended to reduce waiting; the resulting speed and quality still need measurement.

## What a miner receives and returns

The initial task families are `CRAWL` and `EMBED`. A task identifies its inputs, required configuration, expected outputs and applicable assessment criteria. Miners retrieve assigned work and submit outputs with their task and contributor identity.

| Task    | Input                                                                                                | Expected output                                                                                                 |
| ------- | ---------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------- |
| `CRAWL` | Selected URLs, permitted-use instructions, relevant prior source information and extraction settings | Fetch status, permitted page content and extracted text, source URL, observation time and extraction version    |
| `EMBED` | Approved text chunks and their identifiers, a selected model version and representation settings     | Numerical vectors linked to the input chunks in the required order, with the model and configuration identified |

A chunk is a piece of source text prepared for retrieval. An embedding represents that text numerically so search can match meaning, not only exact words. First-phase embedding miners run the selected model; they do not train a replacement or submit vectors from arbitrary models.

The [hardware notice](05-participation-and-progress.md#hardware-planning-for-the-first-phase) sets out GPU needs for embedding miners and validators checking embeddings, and when tested configurations will be published.

## What validators would check

For **crawling**, checks would cover the requested source, fetch outcome, required fields, extraction settings and consistency with permitted source evidence. A changed or inaccessible page must be handled under the task's rules, not automatically treated as misconduct.

For **embeddings**, checks would cover the input identities and order, selected model/configuration, vector count, dimensions and valid numerical values. Validators would use GPU compute to reproduce outputs under the announced checking procedure. Checks must account for permitted numerical variation across supported hardware, rather than assume every valid result is byte-identical.

The first-phase release documentation must specify the actual checking scope, tolerances and handling of unavailable or disputed evidence. A sampled check must be described as sampled; it does not establish that every item was independently verified.

Task assessment and customer deployment remain separate. [Incentives and rewards](04-incentives-and-rewards.md#from-completed-work-to-rewards) explains the proposed path from checked work to weights and rewards; the team would separately approve versions for customer traffic.

## How accepted work becomes searchable

The team would oversee content preparation: filtering low-value content, removing duplicates and splitting accepted text into chunks. Individual operator responsibilities will be specified in the release guide. The collection would retain permitted source text, URLs, observation times and version references independently of embeddings. Replacing a model should not mean losing the information used to create its vectors.

The working design combines keyword retrieval for exact terms with vector retrieval for meaning. Ranking combines and orders candidate results. A search version binds the collection and text preparation to compatible document/query models, indexes and ranking settings. Vectors from unrelated models cannot simply be mixed because their sizes match.

Accepted work enters a candidate version for evaluation, not directly into live search. Before promotion, the team would test useful results, response time, data quality and operating cost. Refreshes and model changes must preserve source/version links and support rollback without restoring deleted material.

## What the release documentation will add

Before requesting participation, the first-phase release must provide a versioned operator guide: installation and configuration, task and submission formats, selected model, tested requirements, validation procedure and transition instructions. Eligibility and reward terms must also be available under the [program-opening requirements](05-participation-and-progress.md#before-a-program-opens).

This overview does not select endpoints, GPU models, batch sizes, reward weights or an activation date. Training, fine-tuning and competitions for better retrieval methods belong to the [wider direction](02-direction-and-first-phase.md#how-the-networks-work-can-expand). They are not part of the initial CRAWL/EMBED tasks.

Continue to [incentives and rewards](04-incentives-and-rewards.md) for how assessed work would count toward rewards.
