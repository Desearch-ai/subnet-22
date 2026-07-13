from types import SimpleNamespace

import numpy as np
import pytest

from desearch.protocol import ResultType, ScraperStreamingSynapse, ScraperTextRole
from neurons.validators.penalty.count_penalty import CountPenaltyModel
from neurons.validators.penalty.date_range_penalty import DateRangePenaltyModel
from neurons.validators.penalty.domain_filter_penalty import DomainFilterPenaltyModel
from neurons.validators.penalty.duplicate_results_penalty import (
    DuplicateResultsPenaltyModel,
)
from neurons.validators.penalty.min_realistic_time_penalty import (
    MinRealisticTimePenaltyModel,
)
from neurons.validators.penalty.miner_score_penalty import MinerScorePenaltyModel
from neurons.validators.penalty.result_schema_penalty import ResultSchemaPenaltyModel
from neurons.validators.penalty.streaming_penalty import StreamingPenaltyModel
from neurons.validators.penalty.summary_structure_penalty import (
    SummaryStructurePenaltyModel,
)
from neurons.validators.penalty.timeout_penalty import TimeoutPenaltyModel
from neurons.validators.reward.content_relevance import ContentRelevanceRewardModel
from neurons.validators.reward.performance_reward import (
    AI_PERF_FLOOR,
    PerformanceRewardModel,
)
from neurons.validators.reward.reward import BaseRewardEvent
from neurons.validators.reward.summary_relevance import SummaryRelevanceRewardModel
from neurons.validators.scrapers.advanced_scraper_validator import (
    AdvancedScraperValidator,
)

WEB_TOOLS = ["Web Search"]
TWITTER_TOOLS = ["Twitter Search"]


def _search_results(n=10):
    return [
        {
            "title": f"Result {i}",
            "link": f"https://example.com/page-{i}",
            "snippet": f"snippet number {i}",
        }
        for i in range(n)
    ]


def _only_links_response(process_time=8.0, tools=WEB_TOOLS):
    resp = ScraperStreamingSynapse(
        prompt="what is bittensor subnet 22",
        tools=tools,
        result_type=ResultType.ONLY_LINKS,
        max_execution_time=15,
        count=10,
        search_results=_search_results(10),
    )
    resp.dendrite.status_code = 200
    resp.dendrite.process_time = process_time
    return resp


def _bad_summary_response(process_time=8.0):
    resp = ScraperStreamingSynapse(
        prompt="what is bittensor subnet 22",
        tools=WEB_TOOLS,
        result_type=ResultType.LINKS_WITH_FINAL_SUMMARY,
        max_execution_time=15,
        count=10,
        search_results=_search_results(10),
    )
    resp.text_chunks = {
        ScraperTextRole.FINAL_SUMMARY.value: ["plain text with no links"]
    }
    resp.dendrite.status_code = 200
    resp.dendrite.process_time = process_time
    return resp


class _StubReward:
    is_deep = True

    def __init__(self, name, values):
        self.name = name
        self.values = np.asarray(values, dtype=np.float32)

    async def apply(self, responses, uids):
        return self.values, {self.name: self.values.tolist()}, [], self.values.tolist()


class _RecordingSub:
    def __init__(self, score):
        self.score = score
        self.seen = []

    async def get_rewards(self, responses, uids):
        self.seen = [r.result_type for r in responses]
        events = [BaseRewardEvent(reward=self.score) for _ in responses]
        return events, [{} for _ in responses]


def _make_validator(monkeypatch, content, summary):
    import neurons.validators.scrapers.base_scraper_validator as bsv

    monkeypatch.setattr(bsv, "build_reward_payload", lambda **k: {})
    monkeypatch.setattr(bsv, "build_log_entry", lambda **k: None)
    monkeypatch.setattr(bsv, "submit_logs_best_effort", lambda *a, **k: None)

    n = len(content)
    v = AdvancedScraperValidator.__new__(AdvancedScraperValidator)
    v.search_type = "ai_search"
    v.content_weight = 0.60
    v.summary_relevance_weight = 0.40
    v.reward_weights = np.array([0.60, 0.40], dtype=np.float32)
    v.component_floors = np.array([0.30, 0.30], dtype=np.float32)
    v.reward_functions = [
        _StubReward("content_stub", content),
        _StubReward("summary_stub", summary),
    ]
    v.penalty_functions = [
        StreamingPenaltyModel(max_penalty=1, neuron=None),
        TimeoutPenaltyModel(max_penalty=1, neuron=None),
        MinRealisticTimePenaltyModel(min_realistic_time=5.0, neuron=None),
        MinerScorePenaltyModel(max_penalty=0.20, neuron=None),
        CountPenaltyModel(max_penalty=1, neuron=None),
        SummaryStructurePenaltyModel(max_penalty=1, neuron=None),
        DuplicateResultsPenaltyModel(max_penalty=1, neuron=None),
        ResultSchemaPenaltyModel(max_penalty=1, neuron=None),
        DateRangePenaltyModel(max_penalty=1, neuron=None),
        DomainFilterPenaltyModel(max_penalty=1, neuron=None),
    ]
    v.performance_model = PerformanceRewardModel(
        neuron=None, min_realistic_time=5.0, target_time=10.0
    )
    v.perf_floor = AI_PERF_FLOOR
    v.log_event = lambda *a, **k: None
    v.neuron = SimpleNamespace(
        config=SimpleNamespace(
            neuron=SimpleNamespace(disable_log_rewards=False), wandb_on=False
        ),
        metagraph=SimpleNamespace(hotkeys=[0] * (n + 1)),
    )
    return v


async def _run(monkeypatch, responses, content, summary):
    v = _make_validator(monkeypatch, content, summary)
    n = len(responses)
    result = await v.compute_rewards_and_penalties(
        event={},
        prompts=["q"] * n,
        responses=responses,
        uids=np.arange(n, dtype=np.int64),
        start_time=0.0,
    )
    return (
        np.asarray(result[0], dtype=np.float32),
        np.asarray(result[5], dtype=np.float32),
        result[3],
    )


@pytest.mark.asyncio
async def test_content_relevance_dispatches_by_tool_not_result_type():
    model = ContentRelevanceRewardModel.__new__(ContentRelevanceRewardModel)
    web_sub = _RecordingSub(0.7)
    tw_sub = _RecordingSub(0.9)
    model.web = web_sub
    model.twitter = tw_sub

    web_resp = _only_links_response(tools=WEB_TOOLS)
    tw_resp = _only_links_response(tools=TWITTER_TOOLS)

    events, _ = await model.get_rewards([web_resp, tw_resp], np.array([0, 1]))

    assert web_sub.seen == [ResultType.ONLY_LINKS]
    assert tw_sub.seen == [ResultType.ONLY_LINKS]
    assert events[0].reward == pytest.approx(0.7)
    assert events[1].reward == pytest.approx(0.9)


@pytest.mark.asyncio
async def test_summary_default_score_rewards_links_only():
    model = SummaryRelevanceRewardModel.__new__(SummaryRelevanceRewardModel)
    score, _, details = await model._default_score(_only_links_response())
    assert score == pytest.approx(1.0)
    assert details["link_count"] == 10


@pytest.mark.asyncio
async def test_summary_default_score_zeros_when_summary_present():
    model = SummaryRelevanceRewardModel.__new__(SummaryRelevanceRewardModel)
    resp = _only_links_response()
    resp.completion = "an unexpected summary"
    score, explanation, _ = await model._default_score(resp)
    assert score == pytest.approx(0.0)
    assert "summary" in explanation.lower()


@pytest.mark.asyncio
async def test_summary_default_score_zeros_without_links():
    model = SummaryRelevanceRewardModel.__new__(SummaryRelevanceRewardModel)
    resp = _only_links_response()
    resp.search_results = []
    score, _, _ = await model._default_score(resp)
    assert score == pytest.approx(0.0)


@pytest.mark.asyncio
@pytest.mark.parametrize("summary_val", [0.0, 1.0])
async def test_only_links_gate_is_content_only(monkeypatch, summary_val):
    weight, gate, _ = await _run(
        monkeypatch, [_only_links_response()], content=[0.60], summary=[summary_val]
    )
    assert gate[0] == pytest.approx(0.60)
    assert weight[0] == pytest.approx(0.60)


@pytest.mark.asyncio
async def test_only_links_content_floor_gates(monkeypatch):
    _, gate, _ = await _run(
        monkeypatch, [_only_links_response()], content=[0.20], summary=[1.0]
    )
    assert gate[0] == pytest.approx(0.0)


@pytest.mark.asyncio
async def test_summary_structure_penalty_skips_only_links_but_fires_on_summary(
    monkeypatch,
):
    responses = [_only_links_response(), _bad_summary_response()]
    _, _, event = await _run(
        monkeypatch, responses, content=[0.60, 0.60], summary=[0.0, 0.0]
    )
    applied = event["summary_structure_penalty_applied"]
    assert applied[0] == pytest.approx(1.0)
    assert applied[1] < 1.0


@pytest.mark.asyncio
async def test_no_penalty_wrongly_fires_on_only_links(monkeypatch):
    _, _, event = await _run(
        monkeypatch, [_only_links_response()], content=[0.60], summary=[0.0]
    )
    for key in (
        "streaming_penalty_applied",
        "summary_structure_penalty_applied",
        "count_penalty_applied",
        "result_schema_penalty_applied",
        "duplicate_results_penalty_applied",
        "timeout_penalty_applied",
        "min_realistic_time_penalty_applied",
        "miner_score_penalty_applied",
        "date_range_penalty_applied",
        "domain_filter_penalty_applied",
    ):
        assert event[key][0] == pytest.approx(1.0)


@pytest.mark.asyncio
async def test_performance_reward_handles_only_links():
    perf = PerformanceRewardModel(neuron=None, min_realistic_time=5.0, target_time=10.0)
    resp = _only_links_response(process_time=8.0)
    assert perf.get_successful_streaming_response(resp)
    events, _ = await perf.get_rewards([resp], np.array([0]))
    assert events[0].reward > 0

    empty = _only_links_response(process_time=8.0)
    empty.search_results = []
    assert perf.get_successful_streaming_response(empty) is None
