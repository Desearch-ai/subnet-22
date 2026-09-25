# Emission

Miners are paid for verified work, in proportion to how much of it they deliver compared with every
other miner. This page explains how that share is worked out and what raises it.

## How weights are set

Every validator checks every upload, and every epoch each validator sets its weights from the
results of its own checks over the last 24 hours. The task API publishes the same figure from the
finalized uploads (`https://task-api.desearch.ai/v1/shares`). A share is your part of all the
verified work of the last 24 hours:

```
your share = rows you were paid for in the last 24 hours / rows everyone was paid for in the last 24 hours
```

Work counts for 24 hours after its upload is finalized. Steady work keeps a steady share; if you stop, your share
fades out over the following day.

A validator sets weights only while it is checking tasks itself. One without a working ScrapingDog
key, or whose checks keep failing, sets none until it recovers.

Validators fetch pages independently, so two honest validators can pay the same upload a few rows
differently when a page changed between their fetches. Their weights then differ by that upload's
share of the day's work, and validator trust on the chain reflects exactly that difference.

## What counts as paid work

**Crawling.** A task you complete is checked by every validator. If it passes, you are paid for
every row at the rate the checked pages matched:

- **pages** you returned count at the share of sampled pages whose text matched the validator's own fetch;
- **pages you reported as failed** count at the share of those the validator could not load either.

A failed task earns nothing. To have a share at all, you must also return at least 85% of the URLs
assigned to you over the same 24 hours.

**Embedding** opens later; see [Embedding tasks](./embedding-tasks.md). There, a passing task is
paid by the characters of the texts it embedded.

## What raises your share

- **Crawl more pages.** Throughput is what you compete on: fetch concurrency, good proxies and
  enough tasks in flight (`MAX_TASKS`).
- **Return the real page.** Text that does not match what a validator fetches lowers what you are paid;
  made-up text fails the task.
- **Load the hard pages.** A page you report as blocked or timed out is paid only if the validator
  cannot load it either. Rotating proxies, or a fallback service such as ScrapingDog, turn those
  pages into paid ones.
- **Finish what you claim.** Return every URL of a task before its claim ends. Expired or abandoned
  tasks count against your coverage and your budget.
- **Stay reliable.** Failed tasks shrink your budget, and repeated failures lock you out of new tasks.

## Your budget

Your budget is how many tasks you may hold at once, from claim until the final verdict. It starts at
1, grows by one for each task paid for at least 85% of its URLs, and halves when a task fails, a claim
expires or you abandon a task. A higher budget lets a fast miner keep more work in flight.

Two failures you caused within 24 hours, if they are at least 5% of your checked tasks, lock you out of
new tasks for 12 hours. Failures on the validator's side, such as its own timeout, never count
against you. Crawling and embedding keep separate budgets and lockouts.

## Checking the numbers yourself

Every finalized upload is public, with the rows it paid and the rows it returned and missed, so a
share can be recomputed from the public list: the rows a miner was paid for in the last 24 hours,
divided by everyone's. What a miner was served, and in which order, is checked against the round's
commitment and signed log with [`task-api/tools/verify_round.py`](../task-api/tools/verify_round.py).
The endpoints are listed in the [task API guide](../task-api/README.md#public-endpoints).
