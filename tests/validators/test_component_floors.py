from types import SimpleNamespace

import numpy as np
import pytest

from desearch.protocol import ResultType


class _StubReward:
    is_deep = True

    def __init__(self, name, values):
        self.name = name
        self.values = np.asarray(values, dtype=np.float32)

    async def apply(self, responses, uids):
        return self.values, {self.name: self.values.tolist()}, [], self.values.tolist()


def _resp(result_type=ResultType.LINKS_WITH_FINAL_SUMMARY):
    return SimpleNamespace(
        result_type=result_type,
        dendrite=SimpleNamespace(process_time=2.0, status_code=200),
        max_execution_time=15,
        mode=None,
    )


def _make_validator(
    monkeypatch, content, summary, weights_matrix=None, component_floors=(0.30, 0.30)
):
    import neurons.validators.scrapers.base_scraper_validator as bsv
    from neurons.validators.scrapers.base_scraper_validator import BaseScraperValidator

    monkeypatch.setattr(bsv, "build_reward_payload", lambda **k: {})
    monkeypatch.setattr(bsv, "build_log_entry", lambda **k: None)
    monkeypatch.setattr(bsv, "submit_logs_best_effort", lambda *a, **k: None)

    n = len(content)
    v = BaseScraperValidator.__new__(BaseScraperValidator)
    v.search_type = "ai_search"
    v.reward_weights = np.array([0.60, 0.40], dtype=np.float32)
    v.component_floors = (
        np.asarray(component_floors, dtype=np.float32)
        if component_floors is not None
        else None
    )
    v.reward_functions = [
        _StubReward("content", content),
        _StubReward("summary", summary),
    ]
    v.penalty_functions = []
    v.performance_model = None
    v.wandb_modality = ""
    v.wandb_reward_keys = []
    v.log_event = lambda *a, **k: None
    v.neuron = SimpleNamespace(
        config=SimpleNamespace(
            neuron=SimpleNamespace(disable_log_rewards=True), wandb_on=False
        ),
        metagraph=SimpleNamespace(hotkeys=[0] * (n + 1)),
    )
    if weights_matrix is not None:
        v.compute_reward_weights_matrix = lambda responses: np.asarray(
            weights_matrix, dtype=np.float32
        )
    return v


async def _run(
    monkeypatch,
    content,
    summary,
    result_types=None,
    weights_matrix=None,
    component_floors=(0.30, 0.30),
):
    v = _make_validator(monkeypatch, content, summary, weights_matrix, component_floors)
    n = len(content)
    result_types = result_types or [ResultType.LINKS_WITH_FINAL_SUMMARY] * n
    responses = [_resp(rt) for rt in result_types]
    result = await v.compute_rewards_and_penalties(
        event={},
        prompts=["q"] * n,
        responses=responses,
        uids=np.arange(n, dtype=np.int64),
        start_time=0.0,
    )
    return np.asarray(result[0], dtype=np.float32), np.asarray(
        result[5], dtype=np.float32
    )


async def _gate(
    monkeypatch,
    content,
    summary,
    result_types=None,
    weights_matrix=None,
    component_floors=(0.30, 0.30),
):
    _, gate = await _run(
        monkeypatch, content, summary, result_types, weights_matrix, component_floors
    )
    return gate


@pytest.mark.asyncio
async def test_low_summary_zeroes_gate_even_when_combined_passes(monkeypatch):
    gate = await _gate(monkeypatch, content=[0.90], summary=[0.20])
    assert gate[0] == pytest.approx(0.0)


@pytest.mark.asyncio
async def test_low_content_zeroes_gate_even_when_combined_passes(monkeypatch):
    gate = await _gate(monkeypatch, content=[0.25], summary=[0.95])
    assert gate[0] == pytest.approx(0.0)


@pytest.mark.asyncio
async def test_floored_response_keeps_its_paid_weight(monkeypatch):
    weight, gate = await _run(monkeypatch, content=[0.90], summary=[0.20])
    assert gate[0] == pytest.approx(0.0)
    assert weight[0] == pytest.approx(0.60 * 0.90 + 0.40 * 0.20)


@pytest.mark.asyncio
async def test_no_component_floors_never_gates(monkeypatch):
    gate = await _gate(
        monkeypatch, content=[0.10], summary=[0.10], component_floors=None
    )
    assert gate[0] == pytest.approx(0.60 * 0.10 + 0.40 * 0.10)


@pytest.mark.asyncio
async def test_both_components_above_floor_keeps_gate(monkeypatch):
    gate = await _gate(monkeypatch, content=[0.90], summary=[0.50])
    assert gate[0] == pytest.approx(0.60 * 0.90 + 0.40 * 0.50)


@pytest.mark.asyncio
async def test_only_links_skips_summary_floor(monkeypatch):
    gate = await _gate(
        monkeypatch,
        content=[0.60],
        summary=[0.0],
        result_types=[ResultType.ONLY_LINKS],
        weights_matrix=[[1.0, 0.0]],
    )
    assert gate[0] == pytest.approx(0.60)


@pytest.mark.asyncio
async def test_floors_applied_per_response(monkeypatch):
    gate = await _gate(monkeypatch, content=[0.90, 0.90], summary=[0.50, 0.10])
    assert gate[0] == pytest.approx(0.60 * 0.90 + 0.40 * 0.50)
    assert gate[1] == pytest.approx(0.0)
