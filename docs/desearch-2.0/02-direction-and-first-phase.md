# Desearch 2.0 — direction and first phase

**Updated:** 2026-09-06

The next step is to build more of what search can prepare in advance: useful source content, searchable representations and a maintained index. That supports the customer needs in our [company foundation](company/company-foundation.md), while creating a base for contributors to improve models and ranking over time.

## Stage of the plan

This is the proposed 2.0 direction, not a launch notice. It does not announce a deployed 2.0 system, completed testnet, active new reward program or open model competition. Progress reports must identify what has actually been built and tested, with dates and evidence.

The first phase prepares the retrieval foundation. Its success depends on useful coverage, supporting text and customer-visible performance; indexing the entire web is not a first-phase promise.

## The first planned work

**Crawl and refresh permitted pages.** Contributors collect useful source content and extracted text so the collection can be maintained.

**Compute embeddings with a selected model.** These representations help retrieval match meaning. This first task runs a specified version of a model on prepared text; it is not model training or invention. Proposing a better model is a separate kind of contribution.

**Check and integrate the work.** Submitted work is assessed, accepted data is assembled into a searchable foundation and the resulting search is tested before release.

The proposed first release places the work queue, stored collection, index and serving system under the team's operation. Miners contribute background work; this design does not promise decentralized storage or live query serving by miners. The [network rationale](company/open-network-and-bittensor.md) explains why SN22 remains central to contribution and improvement even with team-operated serving.

### Building in the background, answering requests

When a customer searches, the proposed system uses a tested version of the prepared index, models and ranking. An ordinary indexed query should not wait for a miner to finish a new task or for a validator decision.

Meanwhile, contributors can refresh sources and produce candidate improvements in the background. Those changes go through assessment and product review before a new version serves customers. Monitoring and rollback are needed to keep a bad update from becoming a lasting customer problem. Moving work off the query path is the design choice; its actual speed and quality still need measurement.

## The first reviewable checkpoint

We propose starting with a bounded source collection and representative queries, not a claim to have indexed the whole web. The first checkpoint should make three things inspectable:

1. The sources and query types selected, and the customer problem that makes them worth testing.
2. A reproducible baseline and evaluation specification, so improvements and regressions can be checked.
3. A preparation decision for the early reranking experiment described below: whether suitable inputs and an evaluator are ready, or what is missing.

The public evidence should cover a few understandable measures:

- **Useful results:** relevant sources and supporting text an application can use.
- **Response time:** retrieval-only time, kept separate from answer generation.
- **Source quality over time:** useful coverage and freshness, not collection size alone.
- **Full cost:** contribution and checking costs, with coordination, integration and serving costs visible too.

This is a proposed checkpoint, not a report that these deliverables are complete or a promise of a launch date. Numeric thresholds and program resources still need to be selected before the relevant experiment. A failed test or an unresolved result is worth reporting when it explains the next decision.

## How the opportunity can grow

| Contribution | Intended opportunity |
|---|---|
| Ranking and reranking | Develop methods that put more useful candidate sources earlier |
| Models and representations | Train or adapt appropriately licensed search models and improve the representations used to retrieve relevant material |
| Data and retrieval methods | Improve useful coverage, freshness or how the system finds candidates |

A bounded reranking experiment within SN22 is the early candidate for testing method improvement. Its sequence and terms remain to be defined; no contest is opened by this proposal. Preparation can proceed alongside the initial build phase when suitable inputs and an evaluator are available; it need not wait for every later capability.

The [company's openness principles](company/open-network-and-bittensor.md) govern adopted model work and its reuse. Each actual program must define what participants submit and how contributions are evaluated and integrated.

Additional storage or serving roles may be explored where they improve outcomes. They are optional possibilities, not a required destination. Future synchronized model training is also not a prerequisite for testing useful independent model contributions.

Continue to [participation and progress](03-participation-and-progress.md).
