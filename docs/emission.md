# Emission

Miners are paid for verified work, in proportion to how much of it they deliver compared with every
other miner. This page explains how that share is worked out and what raises it.

## How weights are set

Every epoch, each validator reads every miner's share from the task API
(`https://task-api.desearch.ai/v1/shares`) and sets its weights by those shares. A share is your part
of all the verified work of the last 24 hours:

```
your share = your credited work in the last 24 hours / everyone's credited work in the last 24 hours
```

Work counts for 24 hours after its verdict. Steady work keeps a steady share; if you stop, your share
fades out over the following day.

## What counts as credited work

**Crawling.** A task you complete is checked by a validator. If it passes, every row is credited at
the rate its sample checked out:

- **pages** you returned count at the share of sampled pages whose text matched the validator's own fetch;
- **pages you reported as failed** count at the share of those the validator could not load either.

A failed task earns nothing. To have a share at all, you must also return at least 85% of the URLs
assigned to you over the same 24 hours.

**Embedding** opens later; see [Embedding tasks](./embedding-tasks.md). There, a passing task is
credited with the characters of the texts it embedded.

## What raises your share

- **Crawl more pages.** Throughput is what you compete on: fetch concurrency, good proxies and
  enough tasks in flight (`MAX_TASKS`).
- **Return the real page.** Text that does not match what a validator fetches lowers your credit;
  made-up text fails the task.
- **Load the hard pages.** A page you report as blocked or timed out is paid only if the validator
  cannot load it either. Rotating proxies, or a fallback service such as ScrapingDog, turn those
  pages into paid ones.
- **Finish what you lease.** Return every URL of a task before its lease ends. Expired or abandoned
  tasks count against your coverage and your budget.
- **Stay reliable.** Failed tasks shrink your budget, and repeated failures lock you out of new tasks.

## Your budget

Your budget is how many tasks you may hold at once, from lease until verdict. It starts at 1, grows
by one for each task credited for at least 85% of its URLs, and halves when a task fails, a lease
expires or you abandon a task. A higher budget lets a fast miner keep more work in flight.

Two failures you caused within 24 hours, if they are at least 5% of your verdicts, lock you out of
new tasks for 12 hours. Failures on the validator's side, such as its own timeout, never count
against you. Crawling and embedding keep separate budgets and lockouts.

## Checking where you stand

```bash
curl -s https://task-api.desearch.ai/v1/miners/<hotkey>          # budget, coverage, verdicts
curl -s "https://task-api.desearch.ai/v1/tasks?miner=<hotkey>"   # your scored tasks
curl -s https://task-api.desearch.ai/v1/tasks/<task_id>          # what the validator found per URL
curl -s https://task-api.desearch.ai/v1/shares                   # every miner's share
```
