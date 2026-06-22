"""Single-item consistency + cost benchmark for the relevance scoring prompt.

Runs the BodyLinkRelevancePrompt (system_body_link_relevance_template from
neurons.validators.utils.prompts) on ONE (query, content) pair RUNS times across
both OpenAI and Chutes scoring models, at the PRODUCTION scoring temperature.
Reports per model:

  - score distribution (counts of scores / other)
  - run-to-run consistency (mode % over RUNS) — the number that matters: the same
    miner link must get the same verdict every time, regardless of which Chutes
    node serves it.
  - input / output tokens and estimated USD cost (OpenAI models only)

Run:  python -m tests.validators.reward.test_consistency_models
Requires OPENAI_API_KEY for OpenAI models, CHUTES_API_TOKEN for Chutes models.
"""

import asyncio
import statistics
import time
from collections import Counter
from typing import List, Union

from desearch.protocol import ScoringModel
from desearch.utils import call_chutes, clean_text, client
from neurons.validators.utils.prompts import BodyLinkRelevancePrompt

RUNS = 100
MODE = "batch"  # "sequential" — one call at a time; "batch" — all in parallel
TEMPERATURE = 0.0001

QUERY = "What are farmers saying about precision agriculture technologies?"

CONTENT = "Precision agriculture demonstrates how data analytics creates tangible value. Farmers require actionable insights on soil health and yield optimization, not complex dashboards. Many teams possess abundant sensor data yet lack the infrastructure for rapid decision-making. Real impact occurs when insights reach stakeholders within minutes. What specific data challenges did Shoshin address? 🎯👀"

# Tested on both cleaned and raw, results are similar.
CONTENT = clean_text(CONTENT)

# OpenAI model names (str) and Chutes models (ScoringModel) can be mixed freely.
MODELS: List[Union[str, ScoringModel]] = [
    ScoringModel.QWEN3_6_27B,
    "gpt-4.1-nano",
]

# Per-1M-token prices in USD (OpenAI only). Update as pricing changes.
PRICING = {
    "gpt-5-nano": {"input": 0.05, "output": 0.40},
    "gpt-4.1-nano": {"input": 0.10, "output": 0.40},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
}


async def run_one(model: Union[str, ScoringModel], prompt: BodyLinkRelevancePrompt) -> dict:
    messages = [
        {"role": "system", "content": prompt.get_system_message()},
        {"role": "user", "content": prompt.text(QUERY, "", "", CONTENT)},
    ]
    started = time.monotonic()

    if isinstance(model, ScoringModel) and model != ScoringModel.OPENAI_GPT4_1_NANO:
        text = await call_chutes(messages, temperature=TEMPERATURE, model=model)
        elapsed = time.monotonic() - started
        score = prompt.extract_score(text) if text else None
        return {
            "score": score,
            "input_tokens": 0,
            "output_tokens": 0,
            "elapsed": elapsed,
        }

    resp = await client.chat.completions.create(
        model=str(getattr(model, "value", model)).replace("openai/", ""),
        messages=messages,
        temperature=TEMPERATURE,
    )
    elapsed = time.monotonic() - started
    text = resp.choices[0].message.content or ""
    return {
        "score": prompt.extract_score(text) if text else None,
        "input_tokens": resp.usage.prompt_tokens,
        "output_tokens": resp.usage.completion_tokens,
        "elapsed": elapsed,
    }


async def run_model(
    model: Union[str, ScoringModel], prompt: BodyLinkRelevancePrompt
) -> List[dict]:
    if MODE == "batch":
        return list(
            await asyncio.gather(*[run_one(model, prompt) for _ in range(RUNS)])
        )
    if MODE == "sequential":
        results: List[dict] = []
        step = max(1, RUNS // 10)
        for i in range(RUNS):
            results.append(await run_one(model, prompt))
            if (i + 1) % step == 0 or (i + 1) == RUNS:
                print(f"  [{model}] {i + 1}/{RUNS}")
        return results
    raise ValueError(f"Unknown MODE: {MODE!r}. Use 'sequential' or 'batch'.")


def report(model: Union[str, ScoringModel], runs: List[dict]) -> None:
    name = str(getattr(model, "value", model))
    scores = [r["score"] for r in runs]
    counts = Counter(s if s is not None else "UNPARSED" for s in scores)
    most, most_n = counts.most_common(1)[0]
    consistency = 100 * most_n / RUNS

    in_tok = sum(r["input_tokens"] for r in runs)
    out_tok = sum(r["output_tokens"] for r in runs)

    print("=" * 90)
    print(f"MODEL  {name}")
    print("-" * 90)
    distribution = ", ".join(f"{s}:{n}" for s, n in counts.most_common())
    print(f"  distribution: {distribution}")
    print(
        f"  mode={most}  consistency={consistency:.1f}%  ({RUNS} runs, temp={TEMPERATURE})"
    )

    times = [r["elapsed"] for r in runs]
    print(
        f"  latency: avg={statistics.mean(times):.3f}s  "
        f"median={statistics.median(times):.3f}s  "
        f"min={min(times):.3f}s  max={max(times):.3f}s"
    )

    price = PRICING.get(name)
    if price and in_tok:
        cost = (in_tok / 1_000_000) * price["input"] + (out_tok / 1_000_000) * price[
            "output"
        ]
        print(f"  tokens:  input={in_tok}  output={out_tok}")
        print(f"  cost:  ${cost:.6f}  total  (~${cost / RUNS:.6f}/call)")


async def main() -> None:
    prompt = BodyLinkRelevancePrompt()
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
            print(f"MODEL  {getattr(model, 'value', model)}")
            print(f"  ERROR: {type(e).__name__}: {e}")


if __name__ == "__main__":
    asyncio.run(main())
