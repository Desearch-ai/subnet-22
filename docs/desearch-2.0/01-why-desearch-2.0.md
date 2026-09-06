# Why Desearch 2.0

**Updated:** 2026-09-06

Our experience operating 1.0 led us to change what the network's work should leave behind: search capabilities we can maintain and improve, not only individual answers.

## What 1.0 achieved

Desearch 1.0 connected a search product with Bittensor's SN22. Miners supplied results using their retrieval approaches, validators evaluated answers and the team operated the API.

Working with miners and validators helped us deliver useful search and learn how to operate a product with outside contributors. Our published evaluations recorded competitive quality alongside response-time weaknesses. The community's work helped establish the foundation for the next version; the evidence below shows both the results and their limits.

## The published benchmark record

We publish the [code and methodology](https://github.com/Desearch-ai/desearch-search-evals), [run files](https://huggingface.co/datasets/desearch/desearch-search-evals/tree/main) and [leaderboard](https://22.desearch.ai) for our AI-search benchmark. As checked on September 6, 2026, the [public record](https://22.desearch.ai/data/latest.json) contains two dated runs: May 31 and July 16, 2026. These are historical results, not a current weekly performance report. The repository's weekly/current-week wording does not match that published record.

The dataset's automatic preview currently reports a generation error. The direct scoreboard and result files linked below remain available; inspecting them does not require the preview to work.

The July run compared five products on 250 same-day news questions. An LLM judge assessed source relevance, answer quality and groundedness.

| July 16, 2026 | Composite quality | Median full-answer time | p90 full-answer time |
|---|---|---|---|
| Desearch | 79.7% | 10.2 s | 12.5 s |
| Exa | 78.1% | 1.6 s | 2.1 s |
| GPT-5-mini | 75.9% | 23.1 s | 46.5 s |
| Tavily | 73.1% | 4.7 s | 6.1 s |
| Perplexity sonar-pro | 68.4% | 5.4 s | 17.5 s |

Values are rounded; p90 uses the nearest-rank 90th percentile of recorded request durations, including timed-out requests. Sources: [July scoreboard](https://huggingface.co/datasets/desearch/desearch-search-evals/raw/main/scoreboards/2026-07-16.json), [per-question results](https://huggingface.co/datasets/desearch/desearch-search-evals/resolve/main/results/2026-07-16.jsonl).

The other recorded model/service labels are `exa-answer`, `tavily-advanced`, `gpt-5-mini` and `perplexity/sonar-pro`; one July timeout row uses the less specific `perplexity` label. Labels do not establish complete historical configurations. The retained result files do not verify every provider's endpoint settings, result budgets or fallback behavior, so the timing comparison must not be read as a matched-configuration test.

Desearch recorded the highest composite in this run, not the highest score on every component. Exa scored higher on source relevance and answer quality; Desearch scored higher on groundedness. The response times measure complete generated answers from the benchmark client. Desearch's median was roughly six times Exa's, about twice Tavily's and Perplexity's, and faster than GPT-5-mini's.

In the May run, Desearch also recorded that run's highest composite: 86.9% against Exa's 86.6%. Median full-answer times were 8.3 seconds and 2.7 seconds respectively. [May scoreboard](https://huggingface.co/datasets/desearch/desearch-search-evals/raw/main/scoreboards/2026-05-31.json), [May results](https://huggingface.co/datasets/desearch/desearch-search-evals/resolve/main/results/2026-05-31.jsonl).

| Run | Recorded Desearch label | Composite weights: source relevance / answer quality / groundedness |
|---|---|---|
| May 31, 2026 | `desearch-miner-deep` | 45% / 25% / 30% |
| July 16, 2026 | `desearch` | 40% / 30% / 30% |

The question sets, weights and recorded labels differ. These runs are not a controlled before-and-after comparison. The [provider module inspected for this review](https://github.com/Desearch-ai/desearch-search-evals/blob/main/providers/desearch.py) describes `LINKS_WITH_FINAL_SUMMARY`, fast mode and 10 results; today's code does not establish both historical configurations. Exact reproduction requires the corresponding code revision, settings and source-time evidence.

These are self-run, news-only, LLM-judged results. They show a bounded quality and waiting-time tradeoff, not a statistically established or independently validated superiority claim. They do not measure retrieval-only latency or Desearch 2.0, establish today's response times, or identify how much delay came from provider calls, miner coordination, answer generation or other processing.

## What operating 1.0 taught us

We saw results supplied through external search providers as well as miners' own retrieval approaches. Answer-level evaluation did not reliably distinguish independently built retrieval capability from externally supplied results. Good answers could therefore depend on capabilities and timing outside our control. That was a limitation of the system and incentives we designed, not a claim that using another provider was misconduct.

We needed more retained source text, supporting passages and reproducible retrieval components to maintain the service as intended. Some capability did accumulate, but depending on an individual miner or provider created a risk: if that service stopped participating, its capability might become unavailable. We are describing an architectural risk, not a specific incident of lost data or coverage.

The response-time gap reinforced the need to inspect and optimize the whole service. Its individual causes were not isolated by the benchmark, so the comparison cannot promise a particular speedup from indexing or from changing the network arrangement.

## The lessons behind the change

| Lesson | Consequence |
|---|---|
| Accuracy and waiting time both determine usefulness | Compare quality and response time on equivalent workloads and service modes |
| Useful answers do not always leave reusable capability | Retain permitted data and adopt methods under rights that allow continued operation and improvement |
| Operational control matters | An index needs maintainable data, components and rights; its ownership does not confer ownership of third-party web content |
| Open contribution can include invention | Create opportunities to improve methods as well as execute prescribed work |
| Correct execution does not establish customer value | Assess work validity, comparative usefulness and deployment separately |
| Customer service must remain dependable | Improvements need an integration process that preserves reliable operation |

A maintained index is Desearch's chosen response to the control and performance problems. External indexes, licensed data, caches and hybrid approaches can also support useful search. The value of our choice must appear in the resulting product.

The [first-phase plan](02-direction-and-first-phase.md) describes how we intend to build on these lessons. [Participation and progress](03-participation-and-progress.md) explains what the proposal means for operators and customers.
