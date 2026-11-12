# Rules

How the task API decides a miner's budget, strikes, take-backs and lockouts, how it cuts rounds into
tasks, and how it counts each miner's work. Every rule reads only what anyone can recompute: chain
blocks, the miner's signed requests, the facts validators report about an uploaded file, and the
miner's on-chain incentive. A replay of the API's inputs with these rules must reach the same budgets,
strikes and accounts. The code is in [`task-api/src/rules.rs`](../task-api/src/rules.rs),
[`clock.rs`](../task-api/src/clock.rs), [`rounds.rs`](../task-api/src/rounds.rs),
[`outages.rs`](../task-api/src/outages.rs) and [`lifecycle.rs`](../task-api/src/lifecycle.rs); every
number below is a setting, named with its default.

## Time

Every deadline, window and lockout is a count of **finalized blocks**. The API reads the chain's
finalized head every 6 seconds and never counts on between reads, so a stalled chain or endpoint
stalls every deadline with it: miners only ever get more time.

Blocks are written as **rule time**, `anchor_at + (block - anchor_block) × 12`, so stored times stay
readable. The anchor is the first finalized block the API read and the wall time it read it at, saved
once and published at `GET /v1/clock` with the current block. A replay turns any stored time back into
its block with it. Wall time is kept only for display and for comparing with storage's own clock.

| Rule | Length |
| --- | --- |
| A claim | `TASK_API_CLAIM_TTL` (180 s, embed `TASK_API_EMBED_CLAIM_TTL` 300 s) plus 10 s of upload grace, rounded up to whole blocks, plus one block for the block the claim was read in. A completion that arrives in the last block counts; the claim is taken back only after it. Miners see the last block as `expires_block` and the wall time it should fall at as `expires_at`. |
| Validators' deadline | 10 blocks for the seed plus `TASK_API_VALIDATION_TTL` (900 s), from the block the completion arrived in |
| An active validator | reported within `TASK_API_ACTIVE_S` (3600 s) |
| Strikes, lockouts, take-backs, shares, the error share | as in the tables below, in rule time |

Signed requests carry their sender's time (`X-Timestamp`), which must be within 60 s of the API's. The
API times them by its host clock only while that clock agrees with the chain's: on every copy, every 15 s
(and on the leading copy every 30 s), it compares its clock with the finalized block's own timestamp
(`Timestamp::Now`) plus 12 s for each block made since. Once the two differ by more than
`TASK_API_CLOCK_SKEW_S` (30 s; 0 turns this off), requests are timed by the chain's clock instead, so a
host clock that jumps never refuses a miner whose own clock is right, and the leading copy opens a
**clock** outage (below).

## Outages

A claim that ends without an upload costs its miner nothing, as when the API restarted while it was
held, when it was alive during an outage the API saw on its own side:

- **storage**: every `TASK_API_STORAGE_PROBE_S` (30 s) the API writes `probe/storage.json` to the uploads
  bucket and reads it back; at once, too, when storage fails a miner's completion. A failed probe opens
  an outage dated from the last probe that worked; the next probe that works closes it.
- **finality**: the finalized block more than `TASK_API_FINALITY_STALL_BLOCKS` (10) behind the best one.
- **clock**: the host clock more than `TASK_API_CLOCK_SKEW_S` (30 s) off the chain's, dated from the last
  check that agreed: requests refused for their timestamp before the check caught it were refused for no
  fault of their miner. The check that agrees again closes it.
- **upload lock**: storage refused to put a completed upload's version under Object Lock, so the completion
  was refused (`TASK_API_UPLOAD_LOCK_S` set); the next lock that holds closes it.

Validators' own word opens one too, whatever the API's probes saw: each validator probes storage the way
miners use it and looks at every completed upload, and reports each failure to the API (signed, logged as a
`witness` entry at the block it arrived). Wherever `TASK_API_STORAGE_WITNESS_QUORUM` (3) distinct validators
reported failures within `TASK_API_STORAGE_WITNESS_WINDOW_S` (300 s) of each other, storage was down from the
first of them to the window's length after the last (`storage_witnessed`). This span is not opened or closed
by the API: it follows from the logged witnesses alone, so a replay finds the same one.

A claim is forgiven when any part of it, from its start to its last block, falls in an outage. Its
receipt names the cause (`restart`, `storage_outage`, `finality_outage`, `clock_outage`, `upload_lock_outage`,
`storage_witnessed_outage`). Outages are listed at `GET /v1/outages`, validators' under `witnessed`, and
logged, so a replay forgives the same lapses.

## Rounds

A crawl round commits to its **URL set** before its seed block exists (`desearch-round-3`):

```
urls_hash     = sha256(canonical_json(sorted [host, url] pairs, by url))
manifest_hash = sha256(canonical_json({"algorithm": "desearch-round-3", "round_id", "seed_block",
                                        "task_urls", "url_count", "urls_hash"}))
```

Only after the seed block's hash gives the seed are the URLs cut into `ceil(url_count / task_urls)`
tasks:

1. hosts in the order of `sha256("{seed}:host:{host}")`, each host's URLs in the order of
   `sha256("{seed}:url:{url}")`;
2. that run dealt across the tasks in turn, so every task gets each host's URLs to within one and
   tasks differ in size by at most one;
3. each task's URLs with their hosts taking turns, in the order each host first appears;
4. task `i` named by the first 16 hex digits of `sha256("{seed}:{round_id}:{i}")`.

The serve order is `serve_order(seed, task_ids)` as before. Nobody, the bot included, chooses which
URLs go together, and every task has the round's mix. `tools/verify_round.py` checks the commitment,
the task names and, given the round's URLs (`--urls`, from `rounds/<round_id>` in the pages bucket),
cuts the round again. Embed rounds still commit to their tasks, one per planner input
(`desearch-serve-order-2`).

With the log anchored on chain (`TASK_API_ANCHOR_NETWORK`), a round is revealed only once an anchor
at or past the log entry holding its commitment was registered in a block before its seed block, so
the API cannot have read the seed before the chain held what it committed to. While an anchor sent
before the seed block may still land, the round waits; once none can, the round commits again to a
later seed block (`recommit`, the same set or tasks under a new `manifest_hash`) and waits for that
one. Each miss leaves the next anchor more room: 30, 70, 150 and 310 blocks, then an epoch, so a
congested chain delays a reveal by about twice its inclusion delay, never for good; periodic anchors
leave 6 of the epoch's commitments to rounds. A replay of the log flags a reveal without such an
anchor, and a commitment made again although one was anchored in time, counting the anchors a
validator's own node shows as well as those the log names.

## Facts and judgments

A validator's report on a crawl upload carries **facts** of the file and the task, the same for every
honest validator: rows returned, duplicate rows, rows for URLs outside the task, error rows, rows
re-extracted from their HTML and those whose text did not come from it, and rows and error rows per host
(`hosts`). Storage answering the pinned entity tag with another version (`upload_changed`) is a fact
too, but it charges nothing: on a bucket without versions the bucket's owner, or a relay that saw the
claim's upload link, can rewrite the key as well as the miner can, and a read by pinned version can't
change at all. Agreeing reports of it make the upload neutral (`upload_changed`), and since the
validators' majority failed it, the miner's next uploads are checked. A report whose facts the task could not have produced (a sampled URL outside the task, more samples
than rows, rows per host that do not add up, …) is refused and recorded as a **wrong report**.

When the row counts could change an upload's accounts (its reports fail it, disagree, fall short of the
quorum, or leave URLs unreturned), the API counts them itself off the version every validator was sent:
rows for the task's URLs, repeats of them, rows for other URLs, and error rows, as the validators'
scoring counts them from the file's `url` and `error` columns. A report whose counts the file
contradicts is left out of the accounts, and when its counts would have charged the miner (more repeats
or other URLs, fewer rows or more error rows than the file has) it is a **wrong report** too. The counts
are logged with the verdict (`recount`, `contradicted`, `wrong_reports`), for anyone holding the public
file to count again and a replay to apply. Facts the file cannot settle (text not from its HTML, a file
one validator could not read) stay as the reports agree on them.

The **live-page comparison** (sampled pages fetched again and compared) is each validator's own
judgment. It decides only what is published and how that validator weights the miner; it never charges
a miner and never pays it in the API's count.

An upload's accounts are judged on its facts:

1. Reports that carry facts (not the validator's own crash or timeout) must agree on every fact. Facts
   two reports disagree on make the upload **neutral** (`facts_disputed`): one report is wrong, and
   anyone can tell which from the public file and the reports.
2. With agreeing facts, `upload_changed` makes it neutral; otherwise the first that applies fails it:
   `unreadable`; `extra_rows` (a duplicate or a URL outside the task); `text_not_from_html` (over 20% of
   re-extracted rows);
   `coverage` (rows returned under the round's gate); `reported_rows` (the miner reported more content
   rows than the file holds, past 2% of the task).
3. A fail charges only when validators of at least `TASK_API_OBJECTIVE_QUORUM` (2) distinct coldkeys
   reported the same facts (a hotkey off the metagraph counts as its own); two hotkeys of one owner are
   one. Below that it is neutral (`below_quorum`): one validator alone never charges a miner. A replay
   counts the owners from the metagraph the log holds, not from the coldkeys the API states.
4. Otherwise it passes.

Reports that fail an upload, or would with one more owner agreeing, are decided only at the upload's
validation deadline: which validators count as active is Redis's to say, not the log's, and a report
still to come could contradict the facts. An upload whose reports pass it may be decided as soon as every
active validator voted. A replay flags an embed fail decided before its deadline; a crawl fail on reports
taken before the deadline settles as neutral (`finalized_early`) when its round closes.

An upload no check drew is judged on the miner's own signed report: it fails `coverage` when its rows
are under the round's gate, else passes. Embed uploads: `vectors_malformed`, `vectors_missing`
and `wrong_size` are facts of the file and fail as above, and `upload_changed` is neutral as above; `vectors_mismatch`, a
validator's own model disagreeing with the miner's vectors, fails only when validators of at least
`TASK_API_EMBED_REFERENCES` (2) distinct coldkeys found it and none passed the vectors; one alone is
neutral (`too_few_references`), and a pass beside a mismatch is neutral (`references_disagree`).
Validators without their own model do not check embed uploads.

| Outcome | Budget | Strike | Count of work | Check record |
| --- | --- | --- | --- | --- |
| pass | grows by half (at least one) when its credit reaches the gate on the URLs outside neutral hosts | — | its credit | a pass |
| fail | halves | yes | minus its URLs outside neutral hosts | a fail: everything since the last pass taken back; the next 10 uploads checked; two among the last 10 take the last day and lock out for 48 h |
| neutral | — | — | — | none; the task goes to another miner |

A crawl upload's row above is applied when its **round closes**, not at its verdict: see below. Until
then its verdict is `held`, its credit 0, and its report shows the round as it stood (`accounts`).

**Credit** (the API's count of work, `GET /v1/shares`): content rows plus error rows at the share of
the miner's failures checks reproduced over 72 h, `(confirmed + 1) / (judged + 2)`, the upload's own
checks included; an embed pass, the characters it was assigned.

**Publishing** follows the validators' majority as before, and only an upload whose accounts did not
fail and whose facts were not disputed is published. A task the majority did not pass goes to another
miner, whatever its accounts.

## Loadability and neutral hosts

For each crawl round the API keeps, from its checked uploads:

- **loadability**: the share of validators' samples whose page the validator could load (`matched`,
  `mismatched`, `errors_unconfirmed` of all samples), once at least `TASK_API_LOADABILITY_SAMPLES` (30)
  were taken;
- per host, the checked uploads with rows of it and those whose rows of it were mostly errors.

The **gate** is `0.85 × loadability` (0.85 until loadability is known). A host is **neutral** for the
round once at least `TASK_API_NEUTRAL_UPLOADS` (3) checked uploads had rows of it and more than
`TASK_API_NEUTRAL_SHARE` (0.5) of them were mostly errors: most miners who got it could not load it. A
neutral host's URLs and rows are left out of every rule and count for everyone that round: no penalty,
no pay. An upload whose rows per host are unknown (unchecked, or an older validator) is scaled to its
task's share of URLs outside neutral hosts. `GET /v1/rounds/{id}` publishes it all as `loadability`.

The gate and the neutral hosts are settled once, when the **round closes** (every task finished or
dropped): each crawl upload of the round is held at its verdict and settled then, in the order its
verdict was finalized, on the round's final loadability and neutral hosts. Its budget, strike, credit,
coverage and check (take-backs, rechecks, the full penalty) are applied in the transaction that closes the
round; what is published, and whether the task goes to another miner, follow the verdict at once. So no
upload is paid for a host that turned neutral after its verdict, or failed on a gate the round later
lowered. The `close` entry of the log states the round's final view (`round`: samples, loadability,
`gate`, hosts, `neutral`) and each upload's accounts (`settled`), and a replay works both out again.
Embed uploads are settled at their verdict, as nothing of theirs depends on the round.

Each validator judges coverage the same way when it weighs miners: on the round's gate and neutral hosts
as the `close` entry of the signed log states them, or, while the round is open, as its own checks of the
round show them (`desearch.credit.round_gate`, `neutral_hosts`), so a host only the owner's miner can
load pays nobody.

## Budget and on-chain incentive

A miner may hold tasks up to `min(budget, cap)`. The cap is the subnet's own verdict on the miner: its
on-chain incentive, set by Yuma consensus from validators' weights, against the incentive its counted
work implies, `0.2 × crawl share + 0.5 × embed share` over the last 24 hours (`TASK_API_CRAWL_EMISSION`,
`TASK_API_EMBED_EMISSION`, as validators split the emission). At or above `TASK_API_INCENTIVE_FULL`
(0.5) of it the cap is the ceiling of 100; below, it falls in proportion, never under 1. A hotkey whose
counted work is younger than `TASK_API_INCENTIVE_GRACE_H` (24) hours, or that is not on the metagraph,
is not capped. `TASK_API_INCENTIVE_FULL=0` turns the cap off.

## Lockouts and strikes

Unchanged except that they count only objective fails and lapses: two strikes within 24 hours, if at
least 5% of the hotkey's judged uploads, lock it out for 1, then 12, then 48 hours within a week;
lapses within 5 minutes of each other are one strike; an upload whose check crashed on every validator
that voted locks it out for a week.

## Validators

A validator is no longer measured against the others. It is refused only while it has sent
`TASK_API_WRONG_REPORTS` (5) or more wrong reports within `TASK_API_WRONG_REPORTS_WINDOW_S` (a day):
reports its task could not have produced, and reports whose row counts the upload's own file contradicts
in the miner's disfavour. How often it agreed with the majority is still shown on its page.
