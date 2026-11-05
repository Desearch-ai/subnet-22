"""How a verdict turns into paid rows: one rule for the task API and every validator."""

from __future__ import annotations

from collections import Counter

SHARE_WINDOW_H = 24
COVERAGE_GATE = 0.85
EVIDENCE = ("matched", "mismatched", "errors_confirmed", "errors_unconfirmed")
RECENT_CHECKS = 10
FAILS_FOR_PENALTY = 2
PENALTY_WINDOW_S = 24 * 3600
# A miner's checks of this long set the rate its unchecked uploads are paid at.
RATE_WINDOW_S = 72 * 3600
# Reported content rows may exceed what a check counts by this share of the task before it fails.
REPORTED_SLACK = 0.02
# A host is neutral for a round, nobody paid or charged for it, once this many checked uploads had rows of it and more than
# this share of them were mostly error rows: most miners who got it could not load it.
NEUTRAL_UPLOADS = 3
NEUTRAL_SHARE = 0.5
# Validators' samples a round needs before its loadability lowers the coverage gate; samples of these outcomes are pages the
# validator could load, whatever the miner got.
LOADABILITY_SAMPLES = 30
LOADED_OUTCOMES = ("matched", "mismatched", "errors_unconfirmed")


def round_gate(sampled: int, loaded: int) -> float:
    """The share of its URLs a task must return: the gate, lowered in proportion when validators could load less of the round."""
    if sampled >= LOADABILITY_SAMPLES and sampled > 0:
        return COVERAGE_GATE * min(1.0, max(0.0, loaded / sampled))
    return COVERAGE_GATE


def covers(returned: int, assigned: int, gate: float) -> bool:
    """Enough rows outside the round's neutral hosts for the task's URLs outside them; a task wholly on neutral hosts always does."""
    return assigned <= 0 or returned >= gate * assigned


def outside_hosts(
    hosts: dict[str, list[int]], assigned: dict[str, int], neutral: set[str]
) -> tuple[int, int]:
    """Rows returned and URLs assigned on hosts that are not neutral."""
    returned = sum(rows for host, (rows, _) in hosts.items() if host not in neutral)
    urls = sum(count for host, count in assigned.items() if host not in neutral)
    return returned, urls


def neutral_hosts(checked: list[dict[str, list[int]]]) -> set[str]:
    """The hosts most of a round's checked uploads could not load, from each upload's rows and error rows per host."""
    uploads: Counter = Counter()
    failed: Counter = Counter()
    for hosts in checked:
        for host, (rows, errors) in hosts.items():
            if rows:
                uploads[host] += 1
                failed[host] += errors * 2 > rows
    return {
        host
        for host, count in uploads.items()
        if count >= NEUTRAL_UPLOADS and failed[host] > NEUTRAL_SHARE * count
    }


def outside_neutral(hosts: dict[str, list[int]], neutral: set[str]) -> float:
    """The share of an upload's rows on hosts that are not neutral; 1 when its rows per host are unknown."""
    total = sum(rows for rows, _ in hosts.values())
    if not total or not neutral:
        return 1.0
    return (
        1.0 - sum(rows for host, (rows, _) in hosts.items() if host in neutral) / total
    )


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


def overstated(reported_ok: int, counted_content: int, assigned: int) -> bool:
    """More content rows in the miner's report than a check counted in its file."""
    return reported_ok > counted_content + max(1.0, REPORTED_SLACK * assigned)
