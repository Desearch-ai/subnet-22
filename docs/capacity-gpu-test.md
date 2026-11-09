# Capacity test: the GPU check

The capacity test is only sound if real GPUs behave as it assumes. Nothing here touches the chain, the task API or miners. It takes
about 4 hours on two rented GPUs, about $15. Run it before the test goes anywhere, even in shadow mode.

## What you need

One RTX 4090 and one H100, each with 100 GB of disk, serving the model as [Miner → Embedding](./miner.md#5-embedding) describes.

```bash
pip install -r requirements.txt
```

## 1. Do more hotkeys earn more on the same GPU?

This is the test's main claim. On each GPU:

```bash
python -m neurons.miners.capacity_bench throughput --url http://127.0.0.1:8000/v1/embeddings \
  --streams 1,2,5,10,38,100 --seconds 60 --out throughput-4090.json
```

**Passes** if `most_above_one_hotkey` is at most 0.02, and `errors` is 0 on every row. Repeat with `--in-flight 4`, `16` and `32`. The
question is whether any setting makes more hotkeys beat one.

**If it fails** batching is giving more hotkeys an edge and the test does not do its job. One hotkey must be able to send as much as
many can, so try a larger `--in-flight`, before anything else.

## 2. How close do honest GPUs get?

Run one server on each GPU, or on two of the same GPU if a different card is not available:

```bash
python -m neurons.miners.capacity_bench compare --url http://gpu-a:8000/v1/embeddings --other-url http://gpu-b:8000/v1/embeddings \
  --count 5000 --real-texts real-texts.txt --out honest.json
```

`real-texts.txt` is real page texts, one a line. Also compare the same GPU with itself under load, and a CPU if you can.

**Passes** if `test_texts` and `real_texts` both have `share_below_match` at 0 and `chance_a_test_passes` at 1. An honest miner must
never fail. Send back `median_gap` and `worst` as well: `within_tight_rule` shows whether the stricter rule in `TIGHT_MEDIAN_GAP`
could be enforced.

## 3. Can a faster, worse model pass?

Serve a cheaper variant, such as 8-bit, FP8, bfloat16 or a quantized build, against the official one:

```bash
python -m neurons.miners.capacity_bench compare --url http://official:8000/v1/embeddings --other-url http://cheap:8000/v1/embeddings \
  --count 5000 --real-texts real-texts.txt --out cheap-fp8.json
python -m neurons.miners.capacity_bench throughput --url http://cheap:8000/v1/embeddings --streams 1 --out cheap-fp8-speed.json
```

Compare the speed with the official model's. **Passes** if a variant that runs faster has `chance_a_test_passes` near 0, or is
outside `within_tight_rule` where honest GPUs from step 2 are inside it. **If it fails**, a faster model passes the check: the threshold
must move before any money depends on it.

## 4. Are test texts a fair stand-in?

Compare characters a second on test texts from step 1 with real tasks on the same GPU, from the embed miner's `task ...: N texts embedded`
log lines. They need not match, since test texts cost less per character than real ones; they should move together between the two
GPUs. **Passes** if the ratio of H100 to 4090 agrees within about 20%.

## 5. How long is a test window really?

Start a stopped model server and time how long it takes to answer its first request, from cold and from a sleeping server if your
version of vLLM offers one. **Report the seconds.** A test lasts about 48 s, so a server that starts in 10 s earns most of a test
without being paid for all day. That decides how real work must count once pay depends on this.

## What to send back

The JSON files and the seconds from step 5. A failure in steps 1 or 2 stops the plan. A failure in step 3 changes the threshold. Steps 4
and 5 inform the rules, and do not stop the shadow run.
