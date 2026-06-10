import asyncio
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

import neurons.validators.scoring.query_scheduler as query_scheduler
from neurons.validators.scoring.constants import (
    COVERAGE_EXPONENT,
    MIN_VOLUME_RATIO,
    QUALITY_EXPONENT,
    QUALITY_THRESHOLDS,
    SEARCH_TYPE_WEIGHTS,
    VOLUME_EXPONENT,
)
from neurons.validators.scoring.query_scheduler import (
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


def test_combine_at_threshold_earns_positive():
    """Raw q (not q_eff): a type exactly at its threshold passes the gate and earns
    q**QUALITY_EXPONENT credit — it is no longer zeroed."""
    out = combine_superlinear_scores(
        {
            "ai_search": {1: (QUALITY_THRESHOLDS["ai_search"], 100)},
            "x_search": {1: (QUALITY_THRESHOLDS["x_search"], 100)},
            "web_search": {1: (QUALITY_THRESHOLDS["web_search"], 100)},
        }
    )
    assert out[1] > 0.0


def test_combine_below_floor_clamps_to_zero():
    out = combine_superlinear_scores(_uniform(0.20, 100))
    assert out[1] == 0.0


def test_combine_volume_is_quadratic():
    """v=100 outearns v=50 by 2^2 = 4× (strong consolidation bonus)."""
    big = combine_superlinear_scores(_uniform(0.6, 100, uid=1))
    small = combine_superlinear_scores(_uniform(0.6, 50, uid=2))
    assert big[1] / small[2] == pytest.approx(2**VOLUME_EXPONENT, rel=0.001)


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
    assert solo[1] / (split[2] + split[3]) == pytest.approx(
        2 ** (VOLUME_EXPONENT - 1), rel=0.001
    )


def test_combine_split_uids_lose_to_higher_quality_solo():
    """1 UID at q=0.5 outearns 2 UIDs at q=0.4 each: the split fails every gate
    (ai<0.45, x/web<0.60) and scores 0, while the solo passes the AI gate."""
    solo = combine_superlinear_scores(_uniform(0.5, 100, uid=1))
    split = combine_superlinear_scores(
        {
            "ai_search": {2: (0.4, 100), 3: (0.4, 100)},
            "x_search": {2: (0.4, 100), 3: (0.4, 100)},
            "web_search": {2: (0.4, 100), 3: (0.4, 100)},
        }
    )
    # split q=0.4 fails every gate (ai<0.45, x/web<0.60) -> 0; solo passes AI.
    assert solo[1] > split[2] + split[3]


def test_combine_quality_gap_amplified_by_raw_q():
    """A 0.10 AI-quality gap produces a raw q**QUALITY_EXPONENT reward gap. Both AI
    qualities clear 0.45 and x/web clear 0.60, so coverage=1 for both (v cancels)."""
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

    def per_type(q_ai, q_x, q_web):
        return (
            SEARCH_TYPE_WEIGHTS["ai_search"] * q_ai**QUALITY_EXPONENT
            + SEARCH_TYPE_WEIGHTS["x_search"] * q_x**QUALITY_EXPONENT
            + SEARCH_TYPE_WEIGHTS["web_search"] * q_web**QUALITY_EXPONENT
        )

    expected = per_type(0.6, 0.7, 0.7) / per_type(0.5, 0.7, 0.7)
    assert a[1] / b[2] == pytest.approx(expected, rel=0.001)


def test_combine_ai_specialist_gets_partial_credit():
    """AI-only at q=1.0, v=100: only AI served, coverage = w_ai."""
    out = combine_superlinear_scores(
        {
            "ai_search": {1: (1.0, 100)},
            "x_search": {1: (0.0, 0)},
            "web_search": {1: (0.0, 0)},
        }
    )
    w = SEARCH_TYPE_WEIGHTS["ai_search"]
    expected = w**COVERAGE_EXPONENT * (w * 1.0**QUALITY_EXPONENT * 100**VOLUME_EXPONENT)
    assert out[1] == pytest.approx(expected)


def test_combine_xweb_specialist_earns_some_not_zero():
    """X+Web at q=1.0 (no AI): coverage = w_x + w_web; earns coverage^(C+1) of a generalist."""
    specialist = combine_superlinear_scores(
        {
            "ai_search": {1: (0.0, 0)},
            "x_search": {1: (1.0, 100)},
            "web_search": {1: (1.0, 100)},
        }
    )
    generalist = combine_superlinear_scores(_uniform(1.0, 100, uid=2))
    cov = SEARCH_TYPE_WEIGHTS["x_search"] + SEARCH_TYPE_WEIGHTS["web_search"]
    expected_ratio = cov**COVERAGE_EXPONENT * cov  # generalist coverage=1, q=v terms cancel
    assert specialist[1] > 0
    assert specialist[1] / generalist[2] == pytest.approx(expected_ratio, rel=0.01)


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
    w = SEARCH_TYPE_WEIGHTS
    coverage = w["ai_search"] * soft_ai + w["x_search"] + w["web_search"]
    per_type = (
        w["ai_search"] * soft_ai * 1.0**QUALITY_EXPONENT * 10**VOLUME_EXPONENT
        + w["x_search"] * 1.0**QUALITY_EXPONENT * 100**VOLUME_EXPONENT
        + w["web_search"] * 1.0**QUALITY_EXPONENT * 100**VOLUME_EXPONENT
    )
    assert out[1] == pytest.approx(coverage**COVERAGE_EXPONENT * per_type)


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
    w = SEARCH_TYPE_WEIGHTS
    expected_per_type = (
        w["ai_search"] * floor_v**VOLUME_EXPONENT
        + w["x_search"] * 100**VOLUME_EXPONENT
        + w["web_search"] * 100**VOLUME_EXPONENT
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
    """1 perfect generalist beats 10 X-only spammers (each only serves X → coverage = w_x)."""
    generalist = combine_superlinear_scores(_uniform(1.0, 100, uid=1))
    x_spam = combine_superlinear_scores(
        {
            "ai_search": {uid: (0.0, 0) for uid in range(2, 12)},
            "x_search": {uid: (1.0, 100) for uid in range(2, 12)},
            "web_search": {uid: (0.0, 0) for uid in range(2, 12)},
        }
    )
    spam_total = sum(x_spam.values())
    wx = SEARCH_TYPE_WEIGHTS["x_search"]
    per_spam = wx**COVERAGE_EXPONENT * wx  # ·100^V cancels vs the generalist
    expected = 1.0 / (10 * per_spam)
    assert generalist[1] / spam_total == pytest.approx(expected, rel=0.02)


def test_combine_perfect_generalist_beats_perfect_specialist():
    """Under weighted coverage^C, a perfect generalist >> an AI-only specialist."""
    generalist = combine_superlinear_scores(_uniform(1.0, 100, uid=1))
    specialist = combine_superlinear_scores(
        {
            "ai_search": {2: (1.0, 100)},
            "x_search": {2: (0.0, 0)},
            "web_search": {2: (0.0, 0)},
        }
    )
    w_ai = SEARCH_TYPE_WEIGHTS["ai_search"]
    assert generalist[1] > specialist[2]
    assert generalist[1] / specialist[2] == pytest.approx(
        1 / (w_ai ** (COVERAGE_EXPONENT + 1)), rel=0.001
    )


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


@pytest.mark.asyncio
async def test_dispatch_epoch_shuffles_uids_before_grouping(monkeypatch):
    epoch_start = datetime(2026, 3, 14, 10, 0, tzinfo=timezone.utc)
    scheduler = QueryScheduler(
        neuron=SimpleNamespace(),
        generator=SimpleNamespace(),
        scoring_store=SimpleNamespace(),
        validators={},
    )

    shuffled_inputs = []

    def reverse_shuffle(uids):
        shuffled_inputs.append(list(uids))
        uids.reverse()

    dispatched_uids = []

    async def dispatch_uid(uid, search_type, uid_items, time_range_start):
        dispatched_uids.append(uid)

    scheduler._dispatch_uid = dispatch_uid
    monkeypatch.setattr(
        QueryScheduler, "_current_hour_start", staticmethod(lambda: epoch_start)
    )
    monkeypatch.setattr(query_scheduler.random, "shuffle", reverse_shuffle)
    monkeypatch.setattr(query_scheduler, "GROUP_SIZE", 2)

    items = [
        {"uid": uid, "search_type": "ai_search", "query": {"query": str(uid)}}
        for uid in [3, 1, 4, 2]
    ]

    await scheduler._dispatch_epoch(items, epoch_start)

    assert [1, 2, 3, 4] in shuffled_inputs
    assert dispatched_uids[:4] == [4, 3, 2, 1]


@pytest.mark.asyncio
async def test_dispatch_uid_sends_half_allocation_per_batch(monkeypatch):
    epoch_start = datetime(2026, 3, 14, 10, 0, tzinfo=timezone.utc)
    scheduler = QueryScheduler(
        neuron=SimpleNamespace(),
        generator=SimpleNamespace(),
        scoring_store=SimpleNamespace(),
        validators={},
    )

    active = 0
    max_active = 0
    total_sent = 0

    async def send_and_save(search_type, uid, query, time_range_start):
        nonlocal active, max_active, total_sent
        active += 1
        total_sent += 1
        max_active = max(max_active, active)
        await asyncio.sleep(0)
        active -= 1

    scheduler._send_and_save = send_and_save
    monkeypatch.setattr(
        QueryScheduler, "_current_hour_start", staticmethod(lambda: epoch_start)
    )
    monkeypatch.setattr(query_scheduler, "BATCH_INTERVAL_SECONDS", 0)

    uid_items = [{"query": {"query": str(i)}} for i in range(100)]

    await scheduler._dispatch_uid(7, "ai_search", uid_items, epoch_start)

    assert total_sent == 100
    assert max_active == 50
