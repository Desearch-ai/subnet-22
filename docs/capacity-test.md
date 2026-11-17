# Capacity test

The capacity test measures how much a miner can embed at one moment. It runs in shadow mode: it sets **no weights and pays
nothing** for now. Anyone can work out the scores, compare them with what they know of their own hardware, and find problems before
any money depends on it.

## Why it exists

Embed tasks are scarcer than miners' capacity, so today each UID gets about the same slice, and income follows the number of UIDs
rather than GPUs. A test taken by every hotkey at the same moment counts a GPU only once, however many hotkeys share it. A coldkey
with the same GPUs scores the same on one UID or on thirty.

## How a test runs

| Step | When | What happens |
| --- | --- | --- |
| Start | Test block `T` | Any block can start a test: its hash decides, about 24 times a day, and nobody knows in advance |
| Work | From `T` | Each hotkey embeds its own texts, in order, as many as it can. The texts come from `T`'s hash and the hotkey, so they cannot be prepared or copied |
| Commit | By block `T + 6` | Each hotkey puts a Merkle root of its vectors and their count on chain with `Commitments.set_commitment`. This costs no fee |
| Sample | Block `T + 10` | That block's hash picks 48 texts per hotkey, after every commitment is final |
| Reveal | By block `T + 35` | Each hotkey publishes those 48 vectors with their proofs |
| Check | After that | Anyone with the model embeds the 48 texts and compares them, at a cosine similarity of at least 0.999 |

A hotkey's score for a test is the characters of all the texts it committed, once its 48 samples pass. A commitment with no reveal,
or a reveal that fails, scores nothing for that test. If over a fifth of the coldkeys that committed have no reveal, the path failed
rather than the miners, and the test counts for nobody. Reports add up a coldkey's hotkeys and name the top one.

The exact rules are in [`desearch/capacity.py`](../desearch/capacity.py).

## Taking part

The embed miner takes the test when you add this to its `.env` and restart it:

```bash
CAPACITY_TEST=1
```

| Setting | Default | |
| --- | --- | --- |
| `CAPACITY_CHAIN` | `finney` | the node to follow. Your own node starts each test a little sooner than a public endpoint |
| `CAPACITY_NETUID` | `22` | the subnet |
| `CAPACITY_IN_FLIGHT` | `4` | batches sent to the model server at once during a test. Raise it if your server has room |
| `CAPACITY_STOP_BLOCKS` | `2` | how many blocks before the deadline to stop and send the commitment, at least 1. Fewer gives more time to embed and more risk of missing the deadline |

While a test runs, real tasks wait between batches, for about 48 seconds. Claims last long enough for this. The miner watches the
newest block, not the finalized one: waiting for finality would lose about 26 seconds of a test.

To use several GPUs under one hotkey, list one server for each in `EMBED_API_URL`, separated by commas; the least busy takes each batch.
A claim with under 2 minutes left keeps going through a test, because losing it costs more than that test's score.

## Scores

Anyone can work out the scores from public data, and it needs nothing from the task API:

```bash
python -m neurons.validators.capacity_score --follow --state capacity.db --out scores.json \
  --model-url http://127.0.0.1:8000/v1/embeddings
```

Without `--model-url` it checks only each reveal's proofs against the chain and credits nothing. A commitment is read as of its test's
deadline block, which a pruned node keeps for about 256 blocks, so keep it following or use an archive node.

Beside each coldkey's share the report shows `service_ratio`, its share of real embed work over its share of the test, and `crawl_share`.
A high score with little real work is capacity that only comes up for tests. Each coldkey's worst test is left out, so one missed
commitment is not a lost day. `above_tight_rule` counts passing tests whose median similarity is further from 1 than honest GPUs are
expected to be: it is reported only, until [the GPU check](./capacity-gpu-test.md) shows where a stricter rule can stand.

## What may change

Shadow mode is here to find these out, so expect them to move before any pay depends on the test: how long a test lasts, how often it
runs, the similarity threshold, and how real tasks count alongside it. Any move to pay will be announced in advance.
