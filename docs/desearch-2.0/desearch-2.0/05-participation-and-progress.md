# Desearch 2.0: participation and progress

## What changes now?

This direction update does not change current weights, reward lanes, registration, hardware requirements or customer endpoints, including existing AI-search, X and social routes. It describes planned work and future compute needs, not an open program. No equipment purchase, new submission or operator change is requested now.

The proposed work is described in [direction and first phase](02-direction-and-first-phase.md). Actual subnet and customer API changes require their own notices.

## How participants' work would change

| Participant               | Proposed change                                                                                                       | Why it matters                                                                   |
| ------------------------- | --------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------- |
| Miners and data operators | Move from supplying search results toward defined work that builds and refreshes retained content and representations | Useful work can serve many later requests instead of ending with one answer      |
| Model builders            | Later opportunities to train or fine-tune retrieval models and develop ranking methods                                | A tested method can improve the product beyond the builder's own requests        |
| Validators and evaluators | Assess the relevant data or model work with task-specific evidence, rather than treating answer quality as sufficient | Credible checks help identify valid contributions and useful improvements        |
| Desearch team             | Integrate accepted work, maintain the collection and operate the initial customer-serving system                      | Keep accountability for product quality, rights, security and dependable service |

These roles describe the direction, not finalized assignments or open work. Initially, running a selected embedding model is an execution task; creating a better model is a separate opportunity.

For how work would count toward rewards, read [incentives and rewards](04-incentives-and-rewards.md). The [contribution roadmap](02-direction-and-first-phase.md#how-the-networks-work-can-expand) explains planned opportunities beyond the first CRAWL/EMBED tasks.

## Hardware planning for the first phase

Embedding miners will need access to GPU-equipped machines for the planned first phase. Validators checking embedding work will also need GPU capacity. The model and tested hardware requirements are not final. Do not buy equipment based on this proposal; wait for the tested requirements and participation notice.

| Work                            | Why compute is needed                                                                         | What not to assume                                                                 |
| ------------------------------- | --------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------- |
| Embedding miners                | Run the selected model on prepared text to generate vectors                                   | This is model execution, not first-phase model training                            |
| Validators checking embeddings  | Run the selected model to reproduce and assess outputs under the published checking procedure | Validator workloads and hardware specifications need not be identical to miners'   |
| Crawling and content extraction | Fetch pages and prepare source-linked text under the task's requirements                      | The embedding GPU requirement is not a blanket requirement for every crawling task |

The first-phase release documentation will include tested minimum and recommended configurations before participation or migration is requested. It will specify supported GPUs and memory, CPU/RAM, storage, network and software requirements for the relevant roles, along with tested workload limits. Exact models, hardware sizes and checking workloads are not specified in this proposal.

Having a GPU alone will not establish eligibility or guarantee rewards. Read the [technical overview](03-first-phase-technical-overview.md) for how the work fits together, and the program-opening requirements below before committing resources.

## What supporters should be able to see

The useful question is what the network is building that people can keep using: maintained information, adopted methods and customer products. Evidence should show useful coverage and data quality, search results and response time, costs, and which contributions reached the system.

A larger index or more participants alone is not proof of commercial success. Nor is immediate search improvement the only possible value of a well-maintained dataset. A future data offering needs evidence for its own intended use.

The [network rationale](../company/open-network-and-bittensor.md#why-bittensor) explains the intended benefit of SN22; [value flows and rights](../company/open-network-and-bittensor.md#value-flows-and-rights) explains its economic boundaries.

## Where your feedback helps now

Which sources and search problems deserve the first bounded tasks? What would make checking contribution quality credible and practical? Which requirements or rights would you need to understand before participating?

This invites input on program design, not investment or work under unspecified terms.

## Before a program opens

Before a program opens, we will publish a usable notice for miners, model builders and validators explaining:

- the work, eligibility, software and measured hardware/operating requirements;
- accepted outputs, evaluation, comparisons and decision responsibilities;
- credit and reward conditions, including rejection, replacement and retirement;
- authorship, retained contributor rights, permitted reuse and intended releases;
- team participation, conflicts and any different eligibility or reward treatment;
- challenge procedures and how changes to terms are communicated.

Execution, comparative evaluation and deployment may have different acceptance criteria. A transition notice must also identify affected operators or customer routes, what carries over, the effective event, support and rollback.

No proposed role establishes permanent rewards or guaranteed profit.

## How we will report progress

We will provide written community updates every week, distinguishing design decisions, tests and live releases. Updates should show what was produced and when, what worked, what failed, which contributions were adopted and what comes next.

The [first-phase checkpoint](02-direction-and-first-phase.md#the-first-reviewable-checkpoint) describes the evidence to prepare. The existing communication cadence is not a promise of weekly releases or a program-opening date.

Operator requirements and reward terms will be communicated before new work is requested. Customer migration, where needed, receives its own notice.

Return to the [reading guide](../README.md).
