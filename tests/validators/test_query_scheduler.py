import asyncio
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

import neurons.validators.scoring.query_scheduler as query_scheduler
from desearch.miner_config import SearchType
from desearch.protocol import SearchMode
from neurons.validators.scoring.constants import (
    MIN_DEEP_SAMPLES_PER_POOL,
    POOL_SHARES,
    QUALITY_EXPONENT,
    VOLUME_EXPONENT,
)
from neurons.validators.scoring.query_scheduler import (
    QueryScheduler,
    combine_pool_scores,
)

AI_FAST = (SearchType.AI_SEARCH, SearchMode.FAST)
AI_DEEP = (SearchType.AI_SEARCH, SearchMode.DEEP)
X_POOL = (SearchType.X_SEARCH, None)


def result(quality, volume, samples=MIN_DEEP_SAMPLES_PER_POOL):
    return (quality, quality, volume, samples)


def test_pool_shares_cover_all_emission():
    assert sum(POOL_SHARES.values()) == pytest.approx(1.0)


def test_pool_pays_exactly_its_share_regardless_of_miner_count():
    for miner_count in (2, 10, 40):
        pool = {uid: result(0.95, 100) for uid in range(miner_count)}
        scores = combine_pool_scores({X_POOL: pool})
        assert sum(scores.values()) == pytest.approx(POOL_SHARES[X_POOL])


def test_splitting_volume_across_uids_loses():
    def pay(uid_count):
        pool = {uid: result(0.80, 600 // uid_count) for uid in range(uid_count)}
        pool[999] = result(0.80, 600)
        scores = combine_pool_scores({AI_FAST: pool})
        return sum(v for uid, v in scores.items() if uid != 999)

    solo, split_2, split_10 = pay(1), pay(2), pay(10)
    assert split_2 < solo
    assert split_10 < split_2


def test_quality_decides_at_equal_volume():
    pool = {0: result(0.90, 100)}
    pool.update({10 + i: result(0.60, 10) for i in range(10)})

    scores = combine_pool_scores({AI_FAST: pool})

    assert scores[0] > sum(scores[10 + i] for i in range(10))


def test_quality_gap_follows_the_quality_exponent():
    pool = {1: result(0.80, 100), 2: result(0.70, 100)}
    scores = combine_pool_scores({AI_FAST: pool})
    assert scores[1] / scores[2] == pytest.approx((0.80 / 0.70) ** QUALITY_EXPONENT)


def test_volume_gap_follows_the_volume_exponent():
    pool = {1: result(0.80, 200), 2: result(0.80, 100)}
    scores = combine_pool_scores({AI_FAST: pool})
    assert scores[1] / scores[2] == pytest.approx(2**VOLUME_EXPONENT)


def test_dominant_miner_takes_the_pool():
    pool = {uid: result(0.90, 100) for uid in range(4)}
    pool[0] = result(0.99, 10_000)

    scores = combine_pool_scores({AI_FAST: pool})

    assert scores[0] > 0.95 * POOL_SHARES[AI_FAST]
    assert sum(scores.values()) == pytest.approx(POOL_SHARES[AI_FAST])


def test_sole_miner_takes_its_whole_pool():
    scores = combine_pool_scores({AI_DEEP: {5: result(0.85, 100)}})
    assert scores[5] == pytest.approx(POOL_SHARES[AI_DEEP])


def test_quality_below_the_gate_earns_nothing():
    assert combine_pool_scores({AI_FAST: {7: result(0.40, 100)}}) == {}


def test_pool_needs_a_minimum_of_deep_samples():
    thin = {AI_FAST: {8: result(0.90, 100, samples=MIN_DEEP_SAMPLES_PER_POOL - 1)}}
    assert combine_pool_scores(thin) == {}


def test_modes_are_scored_as_separate_pools():
    scores = combine_pool_scores(
        {AI_FAST: {1: result(0.90, 100)}, AI_DEEP: {2: result(0.90, 100)}}
    )
    assert scores[1] / scores[2] == pytest.approx(
        POOL_SHARES[AI_FAST] / POOL_SHARES[AI_DEEP]
    )


def test_zero_volume_is_ignored():
    assert combine_pool_scores({AI_FAST: {9: result(0.90, 0)}}) == {}


@pytest.mark.asyncio
async def test_score_epoch_extracts_prompts_from_responses_and_passes_epoch_start():
    scoring_store = SimpleNamespace(
        get_synthetics_for_range=AsyncMock(
            return_value={
                "x_search": [
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
        validators={"x_search": validator},
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
