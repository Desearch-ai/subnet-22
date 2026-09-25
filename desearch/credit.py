"""How a verdict turns into paid rows: one rule for the task API and every validator."""

from __future__ import annotations

from collections import Counter

SHARE_WINDOW_H = 24
COVERAGE_GATE = 0.85
EVIDENCE = ("matched", "mismatched", "errors_confirmed", "errors_unconfirmed")


def credited_urls(ok_rows: int, error_rows: int, outcomes: Counter) -> int:
    """Paid at the sample's rate, so an unsampled forgery still costs."""
    compared = outcomes["matched"] + outcomes["mismatched"]
    judged = outcomes["errors_confirmed"] + outcomes["errors_unconfirmed"]
    pages = round(ok_rows * outcomes["matched"] / compared) if compared else 0
    errors = round(error_rows * outcomes["errors_confirmed"] / judged) if judged else 0
    return pages + errors


def crawl_credit(result: dict) -> int:
    """Rows a crawl verdict pays for: nothing unless it passes on evidence it could read."""
    samples = result.get("samples", [])
    outcomes = Counter(sample["outcome"] for sample in samples)
    if result["verdict"] != "pass" or not any(outcomes[o] for o in EVIDENCE):
        return 0
    if outcomes["unverifiable"] * 2 >= len(samples):
        return 0
    returned, error_rows = result["returned"], result.get("error_rows", 0)
    return credited_urls(returned - error_rows, error_rows, outcomes)


def embed_credit(job: dict, result: dict) -> int:
    """An embed pass is paid the characters it was assigned, or nothing."""
    if result["verdict"] != "pass" or not result.get("matched"):
        return 0
    return int(job.get("chars", 0))
