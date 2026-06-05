from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from neurons.validators.scoring.capacity import QUALITY_THRESHOLDS
from neurons.validators.scoring.query_scheduler import (
    MIN_VOLUME_RATIO,
    QueryScheduler,
    combine_superlinear_scores,
)


def _uniform(q: float, v: int, uid: int = 1) -> dict:
    """Helper: same (q, v) across all three search types for one UID."""
    return {
        "ai_search": {uid: (q, v)},
        "x_search": {uid: (q, v)},
        "web_search": {uid: (q, v)},
    }


def test_combine_at_quality_threshold_earns_zero():
    """At each type's threshold, q_eff = 0 → all per-type contributions zero out."""
    out = combine_superlinear_scores(
        {
            "ai_search": {1: (QUALITY_THRESHOLDS["ai_search"], 100)},
            "x_search": {1: (QUALITY_THRESHOLDS["x_search"], 100)},
            "web_search": {1: (QUALITY_THRESHOLDS["web_search"], 100)},
        }
    )
    assert out[1] == 0.0


def test_combine_below_floor_clamps_to_zero():
    out = combine_superlinear_scores(_uniform(0.20, 100))
    assert out[1] == 0.0


def test_combine_volume_is_mildly_superlinear():
    """v=100 outearns v=50 by 2^1.2 ≈ 2.30× (consolidation bonus)."""
    big = combine_superlinear_scores(_uniform(0.6, 100, uid=1))
    small = combine_superlinear_scores(_uniform(0.6, 50, uid=2))
    assert big[1] / small[2] == pytest.approx(2**1.2, rel=0.001)


def test_combine_solo_beats_same_quality_split():
    """1×(q, 100) outearns 2×(q, 50) at same quality (consolidation bonus, N^(β-1))."""
    solo = combine_superlinear_scores(_uniform(0.6, 100, uid=1))
    split = combine_superlinear_scores(
        {
            "ai_search": {2: (0.6, 50), 3: (0.6, 50)},
            "x_search": {2: (0.6, 50), 3: (0.6, 50)},
            "web_search": {2: (0.6, 50), 3: (0.6, 50)},
        }
    )
    assert solo[1] > split[2] + split[3]
    assert solo[1] / (split[2] + split[3]) == pytest.approx(2**0.2, rel=0.001)


def test_combine_split_uids_lose_to_higher_quality_solo():
    """1 UID at q=0.5 outearns 2 UIDs at q=0.4 each. Under per-type thresholds
    the split's X+Web fail (q<0.50) so they collapse to AI-only coverage=0.6;
    the solo keeps full coverage=1.0 with AI contributing."""
    solo = combine_superlinear_scores(_uniform(0.5, 100, uid=1))
    split = combine_superlinear_scores(
        {
            "ai_search": {2: (0.4, 100), 3: (0.4, 100)},
            "x_search": {2: (0.4, 100), 3: (0.4, 100)},
            "web_search": {2: (0.4, 100), 3: (0.4, 100)},
        }
    )
    assert solo[1] > split[2] + split[3]


def test_combine_quality_gap_amplification_is_quadratic():
    """A 0.10 quality gap produces a (Δq_eff)² reward gap — steep near each
    type's threshold. Both AI qualities are above the 0.45 threshold so both pass."""
    a = combine_superlinear_scores(
        {
            "ai_search": {1: (0.6, 100)},
            "x_search": {1: (0.7, 100)},
            "web_search": {1: (0.7, 100)},
        }
    )
    b = combine_superlinear_scores(
        {
            "ai_search": {2: (0.5, 100)},
            "x_search": {2: (0.7, 100)},
            "web_search": {2: (0.7, 100)},
        }
    )
    ai_thr = QUALITY_THRESHOLDS["ai_search"]
    x_thr = QUALITY_THRESHOLDS["x_search"]
    web_thr = QUALITY_THRESHOLDS["web_search"]
    # Total = coverage² · Σ w·q_eff² (·100^1.2 cancels in ratio). Both have
    # coverage=1 (all 3 served). Compare full per_type_sums.
    def per_type(q_ai, q_x, q_web):
        return (
            0.6 * ((q_ai - ai_thr) / (1 - ai_thr)) ** 2
            + 0.2 * ((q_x - x_thr) / (1 - x_thr)) ** 2
            + 0.2 * ((q_web - web_thr) / (1 - web_thr)) ** 2
        )

    expected = per_type(0.6, 0.7, 0.7) / per_type(0.5, 0.7, 0.7)
    assert a[1] / b[2] == pytest.approx(expected, rel=0.001)


def test_combine_ai_specialist_gets_partial_credit():
    """AI-only at q=1.0, v=100: only AI is served (coverage = 0.60)."""
    out = combine_superlinear_scores(
        {
            "ai_search": {1: (1.0, 100)},
            "x_search": {1: (0.0, 0)},
            "web_search": {1: (0.0, 0)},
        }
    )
    # served = {ai}, coverage = 0.60, per_type = 0.60 * 1.0^2 * 100^1.2
    expected = (0.60**2) * (0.60 * 1.0 * 100**1.2)
    assert out[1] == pytest.approx(expected)


def test_combine_xweb_specialist_earns_some_not_zero():
    """X+Web at q=1.0 (no AI): coverage = 0.40, earns ~6% of perfect generalist."""
    specialist = combine_superlinear_scores(
        {
            "ai_search": {1: (0.0, 0)},
            "x_search": {1: (1.0, 100)},
            "web_search": {1: (1.0, 100)},
        }
    )
    generalist = combine_superlinear_scores(_uniform(1.0, 100, uid=2))
    assert specialist[1] > 0
    assert specialist[1] / generalist[2] == pytest.approx(0.064, abs=0.005)


def test_combine_volume_floor_partial_credit_below_threshold():
    """AI volume below the floor (10% of max=100) gets soft partial credit, not zero."""
    out = combine_superlinear_scores(
        {
            "ai_search": {1: (1.0, 10)},
            "x_search": {1: (1.0, 100)},
            "web_search": {1: (1.0, 100)},
        }
    )
    soft_ai = min(1.0, (10 / 100) / MIN_VOLUME_RATIO)
    coverage = 0.60 * soft_ai + 0.20 * 1.0 + 0.20 * 1.0
    per_type = (
        0.60 * soft_ai * 1.0 * 10**1.2
        + 0.20 * 1.0 * 1.0 * 100**1.2
        + 0.20 * 1.0 * 1.0 * 100**1.2
    )
    assert out[1] == pytest.approx(coverage**2 * per_type)


def test_combine_volume_floor_full_credit_at_minimum():
    """AI volume exactly at the floor (MIN_VOLUME_RATIO of max=100) gets full credit."""
    floor_v = round(MIN_VOLUME_RATIO * 100)
    out = combine_superlinear_scores(
        {
            "ai_search": {1: (1.0, floor_v)},
            "x_search": {1: (1.0, 100)},
            "web_search": {1: (1.0, 100)},
        }
    )
    # All three served at full credit (coverage = 1.0).
    expected_per_type = (
        0.60 * 1.0 * floor_v**1.2 + 0.20 * 1.0 * 100**1.2 + 0.20 * 1.0 * 100**1.2
    )
    assert out[1] == pytest.approx(expected_per_type)


def test_combine_volume_floor_no_cliff():
    """v_ai crossing the floor boundary changes the score smoothly, not 8x."""

    def score_at(v_ai):
        return combine_superlinear_scores(
            {
                "ai_search": {1: (1.0, v_ai)},
                "x_search": {1: (1.0, 100)},
                "web_search": {1: (1.0, 100)},
            }
        )[1]

    b = round(MIN_VOLUME_RATIO * 100)
    below, at, above = score_at(b - 1), score_at(b), score_at(b + 1)
    assert at / below < 1.10  # was 8.46x under the old binary floor
    assert above / at < 1.10


def test_combine_generalist_beats_x_only_spam_team():
    """1 perfect generalist beats 10 X-only spammers by ~12.5×."""
    generalist = combine_superlinear_scores(_uniform(1.0, 100, uid=1))
    x_spam = combine_superlinear_scores(
        {
            "ai_search": {uid: (0.0, 0) for uid in range(2, 12)},
            "x_search": {uid: (1.0, 100) for uid in range(2, 12)},
            "web_search": {uid: (0.0, 0) for uid in range(2, 12)},
        }
    )
    spam_total = sum(x_spam.values())
    assert generalist[1] / spam_total == pytest.approx(12.5, rel=0.05)


def test_combine_perfect_generalist_beats_perfect_specialist():
    """Under 60/20/20 + weighted coverage², generalist >> AI-only specialist."""
    generalist = combine_superlinear_scores(_uniform(1.0, 100, uid=1))
    specialist = combine_superlinear_scores(
        {
            "ai_search": {2: (1.0, 100)},
            "x_search": {2: (0.0, 0)},
            "web_search": {2: (0.0, 0)},
        }
    )
    assert generalist[1] > specialist[2]
    # Generalist earns ~4.6× the AI-only specialist (1.0 / (0.60^2 * 0.60) = 4.63)
    assert generalist[1] / specialist[2] == pytest.approx(1 / (0.6**3), rel=0.001)


@pytest.mark.asyncio
async def test_score_epoch_extracts_prompts_from_responses_and_passes_epoch_start():
    scoring_store = SimpleNamespace(
        get_synthetics_for_range=AsyncMock(
            return_value={
                "web_search": [
                    {
                        "uid": 11,
                        "response": {"query": "what is bittensor", "result": "a"},
                    },
                    {
                        "uid": 12,
                        "response": {"query": "what is tao", "result": "b"},
                    },
                ]
            }
        ),
        get_organics_for_range=AsyncMock(return_value={}),
    )
    validator = SimpleNamespace(compute_rewards_and_penalties=AsyncMock())
    scheduler = QueryScheduler(
        neuron=SimpleNamespace(),
        generator=SimpleNamespace(),
        scoring_store=scoring_store,
        validators={"web_search": validator},
    )
    epoch_start = datetime(2026, 3, 14, 10, 0, tzinfo=timezone.utc)

    await scheduler.score_epoch(epoch_start, allocations_by_type={})

    validator.compute_rewards_and_penalties.assert_awaited_once()
    kwargs = validator.compute_rewards_and_penalties.await_args.kwargs
    assert kwargs["scoring_epoch_start"] == epoch_start
    assert kwargs["prompts"] == ["what is bittensor", "what is tao"]
