"""Single-item consistency + cost benchmark for the relevance scoring prompt.

Runs the original LinkContentPrompt (system_message_question_answer_template
from neurons.validators.utils.prompts) on ONE (query, content) pair RUNS times
across multiple OpenAI models. Reports per model:

  - score distribution (counts of scores / other)
  - run-to-run consistency (mode % over RUNS)
  - total input / output / reasoning tokens
  - estimated USD cost using PRICING below

Run:  python -m tests.validators.reward.test_consistency_models
Requires OPENAI_API_KEY.
"""

import asyncio
import statistics
import time
from collections import Counter
from typing import List

from desearch import client
from desearch.utils import clean_text
from neurons.validators.utils.prompts import LinkContentPrompt

RUNS = 100
MODE = "batch"  # "sequential" — one call at a time; "batch" — all in parallel

QUERY = "What are farmers saying about precision agriculture technologies?"

CONTENT = "Precision agriculture demonstrates how data analytics creates tangible value. Farmers require actionable insights on soil health and yield optimization, not complex dashboards. Many teams possess abundant sensor data yet lack the infrastructure for rapid decision-making. Real impact occurs when insights reach stakeholders within minutes. What specific data challenges did Shoshin address? 🎯👀"

# Tested on both cleaned and raw, results are similar.
CONTENT = clean_text(CONTENT)

MODELS = [
    "gpt-5-nano",
    "gpt-4.1-nano",
    "gpt-4o-mini",
    # "gpt-5.4-nano", more expensive
]

# Per-1M-token prices in USD. Update as OpenAI pricing changes.
PRICING = {
    "gpt-5-nano": {"input": 0.05, "output": 0.40},
    "gpt-4.1-nano": {"input": 0.10, "output": 0.40},
    "gpt-5.4-nano": {"input": 0.20, "output": 1.25},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
}


# Lowest valid reasoning_effort per model. Models not in this map send no
# reasoning_effort at all (gpt-4.x rejects the param outright). The lowest
# accepted value differs across reasoning families:
#   - gpt-5 / 5-mini / 5-nano       -> "minimal"
#   - gpt-5.4 / 5.4-mini / 5.4-nano -> "none"
LOWEST_REASONING_EFFORT = {
    "gpt-5": "minimal",
    "gpt-5-mini": "minimal",
    "gpt-5-nano": "minimal",
    "gpt-5.4": "none",
    "gpt-5.4-mini": "none",
    "gpt-5.4-nano": "none",
}


async def run_one(model: str, prompt: LinkContentPrompt) -> dict:
    kwargs = {
        "model": model,
        "messages": [
            {"role": "system", "content": prompt.get_system_message()},
            {"role": "user", "content": prompt.text(QUERY, CONTENT)},
        ],
        "temperature": 1,
    }
    effort = LOWEST_REASONING_EFFORT.get(model)
    if effort is not None:
        kwargs["reasoning_effort"] = effort

    started = time.monotonic()
    resp = await client.chat.completions.create(**kwargs)
    elapsed = time.monotonic() - started

    text = resp.choices[0].message.content or ""
    score = prompt.extract_score(text) if text else None

    usage = resp.usage
    reasoning_tokens = 0
    details = getattr(usage, "completion_tokens_details", None)

    if details is not None:
        reasoning_tokens = getattr(details, "reasoning_tokens", 0) or 0

    return {
        "score": score,
        "input_tokens": usage.prompt_tokens,
        "output_tokens": usage.completion_tokens,
        "reasoning_tokens": reasoning_tokens,
        "elapsed": elapsed,
    }


async def run_model(model: str, prompt: LinkContentPrompt) -> List[dict]:
    if MODE == "batch":
        return list(
            await asyncio.gather(*[run_one(model, prompt) for _ in range(RUNS)])
        )
    if MODE == "sequential":
        results: List[dict] = []
        step = max(1, RUNS // 10)
        for i in range(RUNS):
            results.append(await run_one(model, prompt))
            done = i + 1
            if done % step == 0 or done == RUNS:
                print(f"  [{model}] {done}/{RUNS}")
        return results
    raise ValueError(f"Unknown MODE: {MODE!r}. Use 'sequential' or 'batch'.")


def report(model: str, runs: List[dict]) -> None:
    scores = [r["score"] for r in runs]
    counts = Counter(s if s is not None else "UNPARSED" for s in scores)
    most, most_n = counts.most_common(1)[0]
    consistency = 100 * most_n / RUNS

    in_tok = sum(r["input_tokens"] for r in runs)
    out_tok = sum(r["output_tokens"] for r in runs)
    reason_tok = sum(r["reasoning_tokens"] for r in runs)

    price = PRICING.get(model, {"input": 0.0, "output": 0.0})
    cost = (in_tok / 1_000_000) * price["input"] + (out_tok / 1_000_000) * price[
        "output"
    ]

    print("=" * 90)
    print(f"MODEL  {model}")
    print("-" * 90)
    distribution = ", ".join(f"{s}:{n}" for s, n in counts.most_common())
    print(f"  distribution: {distribution}")
    print(f"  mode={most}  consistency={consistency:.1f}%  ({RUNS} runs)")
    print(
        f"  tokens:  input={in_tok}  output={out_tok}  "
        f"reasoning={reason_tok}  total={in_tok + out_tok}"
    )
    times = [r["elapsed"] for r in runs]
    print(
        f"  latency: avg={statistics.mean(times):.3f}s  "
        f"median={statistics.median(times):.3f}s  "
        f"min={min(times):.3f}s  max={max(times):.3f}s"
    )
    if model not in PRICING:
        print("  cost:  (no PRICING entry — add one to estimate)")
    else:
        print(f"  cost:  ${cost:.6f}  total  (~${cost / RUNS:.6f}/call)")


async def main() -> None:
    prompt = LinkContentPrompt()
    print(f"Scoring 1 item across {len(MODELS)} model(s), {RUNS} runs each ({MODE}).")
    print(f"Query:    {QUERY}")
    print(f"Content:  {CONTENT[:100]}...")
    print()

    for model in MODELS:
        try:
            runs = await run_model(model, prompt)
            report(model, runs)
        except Exception as e:
            print("=" * 90)
            print(f"MODEL  {model}")
            print(f"  ERROR: {type(e).__name__}: {e}")


if __name__ == "__main__":
    asyncio.run(main())
