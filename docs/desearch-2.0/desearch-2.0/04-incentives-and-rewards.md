# Desearch 2.0: incentives and rewards

Miners would contribute useful data and retrieval capabilities, with assessed work counting toward applicable subnet rewards. This is the proposed idea, not a finalized scoring mechanism. Detailed rules will follow implementation and testing, before participation is requested.

## What the first phase would recognize

The initial programs would assess correct execution of `CRAWL` and `EMBED` tasks. Miners do not need to invent a model for this work.

- **Crawling:** return requested source content and records under the task's rules, not arbitrary page uploads.
- **Embeddings:** compute representations for assigned text using the selected model. GPU ownership alone is not a contribution.

The team would select useful work; validators would assess outputs. See the [technical overview](03-first-phase-technical-overview.md) for the checks.

## From completed work to rewards

The intended path is:

**Published task rules → submitted work → checked contribution records → miner scores → validator weights → protocol reward allocation.**

1. **Know the rules before working.** Each program must state what work is available, who can receive it and what counts as valid completion.
2. **Submit attributable work.** Outputs must identify the miner, task, inputs and required configuration so the submission can be checked.
3. **Assess the contribution.** Validators check the relevant outputs and record the result. The program rules must distinguish valid, invalid and unverifiable work, including source failures and duplicate submissions.
4. **Convert assessed work into scores and weights.** The first-phase rules must define how accepted work is combined and compared across miners and task types. This step is not specified numerically in this proposal.
5. **Apply the protocol.** Eligible validators submit weights: their relative assessments of miners. Bittensor's consensus uses those weights to determine miner reward shares. Validator dividends follow the protocol's own rules, not a Desearch fee for every check. See the [official emissions explanation](https://www.bittensor.com/docs/concepts/emissions#yuma-consensus).

A task score is therefore not a fixed token payment. Completing twice as many tasks does not by itself guarantee twice the rewards. Task eligibility, the scoring rules and protocol allocation all matter.

## A simple example

An embedding miner receives approved text chunks and a specific model, then returns the corresponding vectors. Validators check the submission and reproduce outputs under the announced procedure.

Passing work can qualify for scoring. A batch made with the wrong model or missing required vectors does not meet the same contract. Published rules must determine how failures and disputes are handled; a failed check is not automatically misconduct.

This explains valid execution, not an amount earned. The miner need not develop a better model or personally serve customer queries.

## Rewards and deployment are different decisions

The team separately tests and approves versions for customer use. Correctly completed execution work need not produce an immediate search improvement to qualify under agreed terms.

A deployment decision must not retroactively change promised completion terms. Adoption also does not create permanent rewards or an automatic share of customer revenue.

Validators contribute credible checking, not simply more approvals. Protocol agreement alone does not prove a page is accurate or every output was independently checked.

## How incentives can expand after the first phase

CRAWL and EMBED are the starting task families, not the full extent of future mining work. We plan to expand beyond them. The [contribution roadmap](02-direction-and-first-phase.md#how-the-networks-work-can-expand) describes possible later programs; their sequence and terms are not yet set.

These opportunities need different tests: correct execution for running a selected model; reproducible improvement for a model or reranking competition. Improvement would be tested against a baseline and, where applicable, the starting model. Each program needs announced reward terms; no winner-takes-all or permanent winner policy is selected here.

## When the detailed rules will be available

After implementation and testing, we will publish the scoring rules with worked examples, task-allocation rules, failure and dispute handling, and the mapping from scores to weights. Those details and the effective change notice must be available before participation is requested; they are not selected by this overview.

The [participation notice](05-participation-and-progress.md#before-a-program-opens) sets out the other requirements, including hardware, eligibility and usage rights. No new work or equipment purchase is requested now.
