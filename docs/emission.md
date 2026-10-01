# Emission

Miners are paid for verified work, in proportion to how much of it they deliver compared with every
other miner. This page explains how that share is worked out and what raises it.

## How weights are set

Every validator checks every upload, and every epoch each validator sets its weights from the
results of its own checks over the last 24 hours. A share is your part of all the verified work of
the last 24 hours:

```
your rows = rows you were paid for - URLs of your failed tasks        (never below zero)
your share = your rows / everyone's rows
```

Work counts for 24 hours after its upload is finalized. Steady work keeps a steady share; if you stop, your share
fades out over the following day.

A validator sets weights only while it is checking tasks itself. One without a working ScrapingDog
key, or whose checks keep failing, sets none until it recovers; one with no results of its own yet,
such as right after it starts, puts all its weight on the burn hotkey.

## What counts as paid work

**Crawling.** A task you complete is checked by every validator. If it passes, you are paid for
every row at the rate the checked pages matched:

- **pages** you returned count at the share of sampled pages whose text matched the validator's own fetch;
- **pages you reported as failed** count at the share of those the validator could not load either.

A failed task earns nothing, and its URLs are taken back from the rows you are paid for in the same
24 hours.

**Embedding** opens later; see [Embedding tasks](./embedding-tasks.md). There, a passing task is
paid by the characters of the texts it embedded.

## What raises your share

- **Crawl more pages.** Throughput is what you compete on: fetch concurrency and good proxies. One
  fast miner can hold as much work as several slow ones.
- **Return the real page.** Text that does not match what a validator fetches lowers what you are paid;
  made-up text fails the task.
- **Load the hard pages.** A page you report as blocked or timed out is paid only if the validator
  cannot load it either. Rotating proxies, or a fallback service such as ScrapingDog, turn those
  pages into paid ones.
- **Claim only what you can finish.** Upload each task within 3 minutes of claiming it. An expired
  or handed-back claim costs its URLs, halves your budget and is a strike. The reference miner
  claims only what its measured pace can finish.
- **Stay reliable.** Failed tasks shrink your budget, and repeated failures lock you out of new tasks.

## Your budget

Your budget is how many tasks you may crawl at once.

- It starts at 1, grows by half (at least one) for each task paid for at least 85% of its URLs, up
  to 100, and halves when a task fails, a claim expires or you hand a task back.
- Up to twice your budget in uploads may wait for a verdict while you keep crawling.
- One claim returns as many tasks as you ask for and your budget allows; ask at most once every
  2 seconds.

A strike is a failure you caused, or a claim that expired or you handed back; claims that lapse
within 5 minutes of each other count as one. Two strikes within 24 hours, if they are at least 5% of
your checked tasks, lock you out of new tasks for an hour, then 12 hours, then 48 within a week. An
upload that crashes most validators' checks locks you out for a week. Failures on the validator's
side never count against you. Crawling and embedding keep separate budgets and lockouts.

## Checking the numbers yourself

Every finalized upload is public, with the rows it paid and the rows it returned and missed, so a
share can be recomputed from the public list as above. What a miner was served, and in which order,
is checked against the round's commitment and signed log with [`task-api/tools/verify_round.py`](../task-api/tools/verify_round.py).
The endpoints are listed in the [task API guide](../task-api/README.md#public-endpoints).
