from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock

import numpy as np
import pytest

import neurons.validators.scoring.query_scheduler as query_scheduler
from neurons.validators.scoring.query_scheduler import QueryScheduler
from neurons.validators.scrapers.base_scraper_validator import BaseScraperValidator


class _Penalty:
    is_deep = False

    def __init__(self, applied):
        self.applied = np.asarray(applied, dtype=np.float32)

    async def apply_penalties(self, responses, uids, params):
        return self.applied, self.applied, self.applied


class _DeepReward:
    is_deep = True
    name = "deep"

    async def apply(self, responses, uids):
        raise AssertionError("deep reward must not run on the cheap path")


def _validator(penalty_functions):
    v = BaseScraperValidator.__new__(BaseScraperValidator)
    v.search_type = "x_search"
    v.reward_weights = np.array([1.0], dtype=np.float32)
    v.reward_functions = [_DeepReward()]
    v.penalty_functions = penalty_functions
    return v


@pytest.mark.asyncio
async def test_cheap_returns_one_when_no_penalty_fires():
    responses = [{"a": 1}, {"a": 2}]
    uids = np.array([1, 1], dtype=np.int64)

    v = _validator([_Penalty([1.0, 1.0])])
    out = await v.compute_cheap_scores(responses, uids)

    assert out.tolist() == [1.0, 1.0]


@pytest.mark.asyncio
async def test_cheap_drops_below_one_when_penalty_fires_and_never_adds_reward():
    responses = [{"a": 1}, {"a": 2}]
    uids = np.array([1, 1], dtype=np.int64)

    v = _validator([_Penalty([0.5, 1.0])])
    out = await v.compute_cheap_scores(responses, uids)

    assert out.tolist() == [0.5, 1.0]
    assert float(out.max()) <= 1.0


@pytest.mark.asyncio
async def test_cheap_multiplies_multiple_penalties():
    responses = [{"a": 1}]
    uids = np.array([1], dtype=np.int64)

    v = _validator([_Penalty([0.5]), _Penalty([0.4])])
    out = await v.compute_cheap_scores(responses, uids)

    assert out[0] == pytest.approx(0.2)


def _scheduler(validator):
    return QueryScheduler(
        neuron=SimpleNamespace(),
        generator=SimpleNamespace(),
        scoring_store=SimpleNamespace(),
        validators={"x_search": validator},
    )


def _synth(uid, n, prefix=""):
    return [
        {"uid": uid, "response": {"query": f"{prefix}q{i}", "uid": uid}}
        for i in range(n)
    ]


async def _score(scheduler, synth_items, deep_score, cheap_multiplier, monkeypatch):
    validator = scheduler.validators["x_search"]

    async def fake_full(self, validator_arg, items, time_range_start):
        arr = np.full(len(items), deep_score, dtype=np.float32)
        return arr, arr.copy()

    monkeypatch.setattr(QueryScheduler, "_run_full_scoring", fake_full)

    async def fake_cheap(responses, uids):
        return np.full(len(responses), cheap_multiplier, dtype=np.float32)

    validator.compute_cheap_scores = fake_cheap
    monkeypatch.setattr(query_scheduler.capacity, "record_window_quality", AsyncMock())

    return await scheduler._score_one_type(
        search_type="x_search",
        synthetics={"x_search": synth_items},
        organics={},
        time_range_start=datetime(2026, 3, 14, 10, 0, tzinfo=timezone.utc),
        window_start="2026-03-14T10:00:00+00:00",
        allocations_by_lane={},
    )


def _uid(out, uid):
    return out[None][uid]


@pytest.mark.asyncio
async def test_clean_cheap_leaves_quality_at_deep_mean(monkeypatch):
    validator = SimpleNamespace()
    scheduler = _scheduler(validator)

    out = await _score(scheduler, _synth(1, 5), 0.8, 1.0, monkeypatch)

    _q_gate, q_weight, _vol, _samples = _uid(out, 1)
    assert q_weight == pytest.approx(0.8)


@pytest.mark.asyncio
async def test_failing_cheap_penalty_lowers_quality(monkeypatch):
    clean = await _score(
        _scheduler(SimpleNamespace()), _synth(1, 5), 0.8, 1.0, monkeypatch
    )
    penalized = await _score(
        _scheduler(SimpleNamespace()), _synth(1, 5), 0.8, 0.5, monkeypatch
    )

    assert _uid(penalized, 1)[1] < _uid(clean, 1)[1]
    assert _uid(penalized, 1)[1] == pytest.approx(0.8 * 0.5)


@pytest.mark.asyncio
async def test_more_clean_cheap_items_do_not_raise_quality(monkeypatch):
    few = await _score(
        _scheduler(SimpleNamespace()), _synth(1, 5), 0.8, 1.0, monkeypatch
    )
    many = await _score(
        _scheduler(SimpleNamespace()), _synth(1, 50), 0.8, 1.0, monkeypatch
    )

    assert _uid(many, 1)[1] == pytest.approx(_uid(few, 1)[1])
    assert _uid(many, 1)[2] >= _uid(few, 1)[2]
