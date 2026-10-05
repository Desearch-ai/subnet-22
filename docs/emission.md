# Emission

Miners are paid for verified work, in proportion to how much of it they deliver compared with every
other miner. This page explains how that share is worked out and what raises it.

## How weights are set

Every validator works out the weights itself, from the public uploads bucket. It reads the log of
every completed upload there, with the counts each miner reported, and checks a random sample of
them. It assumes the sample is representative of the miner's other uploads: work it did not check
is paid at the rate its checks of that miner paid.

```
your rows  = rows your checked uploads paid
           + your other uploads' reported rows × the rate your checks paid over the last 3 days
           - URLs of your failed checks
           - what a failed check takes back                                   (never below zero)
your share = your rows / everyone's rows
```

Uploads count for 24 hours after they are completed. Steady work keeps a steady share; if you stop,
your share fades out over the following day.

A validator sets weights only while it is checking tasks itself. One without a working ScrapingDog
key, or whose checks keep failing, sets none until it recovers.

## What counts as paid work

**Crawling.** Every validator checks a random sample of your uploads, and every upload of a new
hotkey until it has passed 10 checks. A checked task that passes pays every row at the rate its
checked pages matched: pages at the share whose text matched the validator's own fetch, and pages
you reported as failed at the share of those the validator could not load either. Your other
uploads are paid on the counts you reported, at the rate your checks paid. A report under 85% of
the task's URLs fails.

A failed check earns nothing and costs the task's URLs. It also takes back everything you uploaded
since your last passed check, and those pages leave the published set; your next 10 uploads are
all checked. Two failed checks among your last 10 take back your last 24 hours and lock you out for
48 hours. A checked task whose report claims more pages than its file holds fails.

**Embedding** opens later; see [Embedding tasks](./embedding-tasks.md). There, a passing task is
paid by the characters of the texts it embedded.

## What raises your share

- **Crawl more pages.** Throughput is what you compete on: fetch concurrency and good proxies. One
  fast miner can hold as much work as several slow ones.
- **Return the real page.** Text that does not match what a validator fetches lowers what you are paid;
  made-up text fails the task, and a failed check takes back your unchecked work since your last
  passed one.
- **Report your counts truthfully.** Unchecked uploads are paid on them, and checked ones are held
  to them.
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
