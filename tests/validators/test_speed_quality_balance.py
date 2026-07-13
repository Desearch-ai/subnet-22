"""Speed-vs-quality balance, driving the real scoring functions with mocked
relevance. GATE (served + capacity ramp) = pure quality x penalties; WEIGHT
(emissions rank) = quality x perf_factor x penalties; perf_floor 0.50 -> fast+good
earns ~8x a same-quality slow answer while the slow answer keeps its capacity.

Archetypes on a FAST query (perf target 5s, serving allowance 15s):
  A slow+good (no index, 14s) | B fast+good (cached, 2.5s) | C fast+trash (4s)
  D canned (0.8s)             | E good but overdue (18s, past the 15s allowance)
"""

import asyncio
from types import SimpleNamespace

import numpy as np
import pytest

from desearch.protocol import SearchMode
from neurons.validators.penalty.min_realistic_time_penalty import (
    MinRealisticTimePenaltyModel,
)
from neurons.validators.penalty.timeout_penalty import TimeoutPenaltyModel
from neurons.validators.reward.performance_reward import (
    PerformanceRewardModel,
    perf_factor,
)
from neurons.validators.scoring.query_scheduler import combine_superlinear_scores

SERVING_ALLOWANCE = 15
AI_PERF_FLOOR = 0.50
AI_THRESHOLD = 0.50

_perf = PerformanceRewardModel(neuron=None)
_min_real = MinRealisticTimePenaltyModel(neuron=None)
_timeout = TimeoutPenaltyModel(max_penalty=1, neuron=None)


def _mock_response(process_time, mode=SearchMode.FAST):
    return SimpleNamespace(
        dendrite=SimpleNamespace(process_time=process_time, status_code=200),
        max_execution_time=SERVING_ALLOWANCE,
        mode=mode,
        timeout=SERVING_ALLOWANCE + 5,
    )


def _quality(content, summary):
    return 0.625 * content + 0.375 * summary


async def _penalty_mult(resp):
    mult = 1.0
    for pen in (_min_real, _timeout):
        _, _, applied = await pen.apply_penalties([resp], [0], None)
        mult *= float(applied[0])
    return mult


async def _scores(content, summary, process_time):
    resp = _mock_response(process_time)
    quality = _quality(content, summary)
    penalty = await _penalty_mult(resp)
    perf_raw = _perf.reward(process_time, _perf._scoring_budget(resp))
    pf = perf_factor(perf_raw, AI_PERF_FLOOR)
    return quality * penalty, quality * pf * penalty


ARCHETYPES = {
    "A_slow_good_14s": (0.90, 0.88, 14.0),
    "B_fast_good_2.5s": (0.90, 0.88, 2.5),
    "C_fast_trash_4s": (0.10, 0.10, 4.0),
    "D_canned_0.8s": (0.90, 0.88, 0.8),
    "E_good_overdue_18s": (0.90, 0.88, 18.0),
}


async def _all():
    return {k: await _scores(*v) for k, v in ARCHETYPES.items()}


@pytest.mark.asyncio
async def test_weight_ordering_fast_beats_slow_beats_trash():
    s = {k: w for k, (g, w) in (await _all()).items()}
    assert s["B_fast_good_2.5s"] > s["A_slow_good_14s"]
    assert s["A_slow_good_14s"] > s["C_fast_trash_4s"]
    assert s["E_good_overdue_18s"] > s["C_fast_trash_4s"]
    assert s["D_canned_0.8s"] == 0.0
    assert s["B_fast_good_2.5s"] / s["A_slow_good_14s"] == pytest.approx(
        1 / AI_PERF_FLOOR, rel=1e-3
    )


@pytest.mark.asyncio
async def test_gate_keeps_slow_good_alive_and_blocks_trash():
    g = {k: gate for k, (gate, w) in (await _all()).items()}
    assert g["A_slow_good_14s"] >= AI_THRESHOLD
    assert g["B_fast_good_2.5s"] >= AI_THRESHOLD
    assert g["C_fast_trash_4s"] < AI_THRESHOLD
    assert g["D_canned_0.8s"] < AI_THRESHOLD


@pytest.mark.asyncio
async def test_emissions_fast_earns_about_8x_slow_same_quality():
    a = await _all()
    gB, wB = a["B_fast_good_2.5s"]
    gA, wA = a["A_slow_good_14s"]
    gC, wC = a["C_fast_trash_4s"]
    combined = combine_superlinear_scores(
        {"ai_search": {0: (gB, wB, 10), 1: (gA, wA, 10), 2: (gC, wC, 10)}}
    )
    assert combined[1] > 0
    assert combined.get(2, 0.0) == 0.0
    assert combined[0] / combined[1] == pytest.approx(
        (1 / AI_PERF_FLOOR) ** 3, rel=1e-2
    )


def test_volume_exponent_is_anti_sybil():
    solo = combine_superlinear_scores({"ai_search": {0: (0.7, 100)}})
    split = combine_superlinear_scores({"ai_search": {1: (0.7, 50), 2: (0.7, 50)}})
    assert solo[0] > split[1] + split[2]


@pytest.mark.asyncio
async def test_canning_not_rewarded_over_real_work_in_deep_mode():
    assert _perf.reward(2.1, 30) == _perf.reward(18.0, 30) == 1.0


class _StubQuality:
    is_deep = True
    name = "content_relevance"

    def __init__(self, values):
        self.values = np.asarray(values, dtype=np.float32)

    async def apply(self, responses, uids):
        return self.values, {self.name: self.values.tolist()}, [], self.values.tolist()


class _StubPerf:
    def __init__(self, perf_raws):
        self.perf_raws = perf_raws

    async def get_rewards(self, responses, uids):
        return [SimpleNamespace(reward=r) for r in self.perf_raws], {}


@pytest.mark.asyncio
@pytest.mark.parametrize("floor", [0.50, 0.70])
async def test_compute_rewards_real_composition(monkeypatch, floor):
    import neurons.validators.scrapers.base_scraper_validator as bsv
    from neurons.validators.scrapers.base_scraper_validator import BaseScraperValidator

    monkeypatch.setattr(bsv, "build_reward_payload", lambda **k: {})
    monkeypatch.setattr(bsv, "build_log_entry", lambda **k: None)
    monkeypatch.setattr(bsv, "submit_logs_best_effort", lambda *a, **k: None)

    quality = [0.90, 0.90]
    perf_raws = [1.0, 0.0]
    responses = [_mock_response(2.5, mode=None), _mock_response(14.0, mode=None)]

    v = BaseScraperValidator.__new__(BaseScraperValidator)
    v.search_type = "ai_search"
    v.reward_weights = np.array([1.0], dtype=np.float32)
    v.reward_functions = [_StubQuality(quality)]
    v.penalty_functions = [_min_real, _timeout]
    v.performance_model = _StubPerf(perf_raws)
    v.perf_floor = floor
    v.wandb_modality = ""
    v.wandb_reward_keys = []
    v.log_event = lambda *a, **k: None
    v.neuron = SimpleNamespace(
        config=SimpleNamespace(
            neuron=SimpleNamespace(disable_log_rewards=True), wandb_on=False
        ),
        metagraph=SimpleNamespace(hotkeys=[0, 0]),
    )

    result = await v.compute_rewards_and_penalties(
        event={},
        prompts=["q", "q"],
        responses=responses,
        uids=np.array([0, 1], dtype=np.int64),
        start_time=0.0,
    )
    weight, gate = np.asarray(result[0]), np.asarray(result[5])

    assert gate[0] == pytest.approx(0.90)
    assert gate[1] == pytest.approx(0.90)
    assert weight[0] == pytest.approx(0.90)
    assert weight[1] == pytest.approx(0.90 * floor)
    assert gate[1] >= weight[1]


def test_combine_gate_decides_served_weight_decides_credit():
    good_slow = combine_superlinear_scores({"ai_search": {0: (0.85, 0.45, 50)}})
    assert good_slow[0] > 0
    fast = combine_superlinear_scores({"ai_search": {0: (0.85, 0.85, 50)}})[0]
    assert fast / good_slow[0] == pytest.approx((0.85 / 0.45) ** 3, rel=1e-6)
    trash = combine_superlinear_scores({"ai_search": {0: (0.40, 0.95, 50)}})
    assert trash.get(0, 0.0) == 0.0


def test_mode_synapse_roundtrip():
    from desearch.protocol import ResultType, ScraperStreamingSynapse

    syn = ScraperStreamingSynapse(
        prompt="q",
        max_execution_time=15,
        mode=SearchMode.FAST,
        result_type=ResultType.LINKS_WITH_FINAL_SUMMARY,
    )
    assert _perf._scoring_budget(syn) == 5.0
    syn2 = ScraperStreamingSynapse(
        prompt="q",
        max_execution_time=15,
        result_type=ResultType.LINKS_WITH_FINAL_SUMMARY,
    )
    assert _perf._scoring_budget(syn2) == 15.0


def _print_table():
    a = asyncio.get_event_loop().run_until_complete(_all())
    print("\n=== per-response (fast query: target 5s, allowance 15s, floor 0.50) ===")
    print(f"  {'archetype':22s} {'gate(pure)':>11} {'weight(rank)':>13}")
    for k in sorted(a, key=lambda x: -a[x][1]):
        g, w = a[k]
        print(f"  {k:22s} {g:>11.4f} {w:>13.4f}")
    gB, wB = a["B_fast_good_2.5s"]
    gA, wA = a["A_slow_good_14s"]
    gC, wC = a["C_fast_trash_4s"]
    combined = combine_superlinear_scores(
        {"ai_search": {0: (gB, wB, 10), 1: (gA, wA, 10), 2: (gC, wC, 10)}}
    )
    print("\n=== emissions (combine, equal volume 10) ===")
    print(f"  B fast+good : {combined.get(0, 0):.3f}")
    print(f"  A slow+good : {combined.get(1, 0):.3f}")
    print(f"  C fast+trash: {combined.get(2, 0):.3f}")
    print(f"  -> fast/slow emissions ratio = {combined[0] / combined[1]:.1f}x")


if __name__ == "__main__":
    _print_table()
