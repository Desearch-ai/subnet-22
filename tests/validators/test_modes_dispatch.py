import random
from collections import Counter
from types import SimpleNamespace

import pytest

from desearch.protocol import ResultType
from desearch.utils import (
    SERVING_FLOOR,
    SearchMode,
    MODE_BUDGETS,
    get_mode_budget,
    get_mode_serving_budget,
)
from neurons.validators.penalty.min_realistic_time_penalty import (
    MinRealisticTimePenaltyModel,
)
from neurons.validators.reward.performance_reward import PerformanceRewardModel
from neurons.validators.scoring.synthetic_query_generator import (
    TWITTER_TOOL,
    WEB_TOOL,
    pick_ai_mode_and_tool,
    random_result_types,
)


def test_mode_budget_map():
    assert MODE_BUDGETS == {"fast": 5, "balanced": 15, "deep": 30}
    assert get_mode_budget("fast") == 5
    assert get_mode_budget("balanced") == 15
    assert get_mode_budget("deep") == 30


def test_mode_enum_interop():
    assert get_mode_budget(SearchMode.DEEP) == 30
    assert get_mode_budget("fast") == 5
    with pytest.raises(ValueError):
        get_mode_budget("nonsense")


def test_serving_budget_floors_fast_only():
    assert get_mode_serving_budget(SearchMode.FAST) == SERVING_FLOOR == 15
    assert get_mode_serving_budget("fast") > get_mode_budget("fast")
    assert get_mode_serving_budget("balanced") == get_mode_budget("balanced") == 15
    assert get_mode_serving_budget("deep") == get_mode_budget("deep") == 30


def test_per_query_mix_single_tool_any_mode():
    seen_modes = set()
    seen_tools = set()
    fast_tools = set()

    for _ in range(2000):
        mode, tools = pick_ai_mode_and_tool()

        assert len(tools) == 1
        assert tools[0] in (WEB_TOOL, TWITTER_TOOL)

        seen_modes.add(mode)
        seen_tools.add(tools[0])
        if mode == "fast":
            fast_tools.add(tools[0])

    assert seen_modes == {"fast", "balanced", "deep"}
    assert seen_tools == {WEB_TOOL, TWITTER_TOOL}
    assert fast_tools == {WEB_TOOL, TWITTER_TOOL}


def test_per_query_result_type_mix():
    counts = Counter(random_result_types)
    assert counts[ResultType.LINKS_WITH_FINAL_SUMMARY] == 4
    assert counts[ResultType.ONLY_LINKS] == 1

    seen = {random.choice(random_result_types) for _ in range(2000)}
    assert seen == {ResultType.LINKS_WITH_FINAL_SUMMARY, ResultType.ONLY_LINKS}


def _make_perf_model():
    return PerformanceRewardModel.__new__(PerformanceRewardModel)


def test_perf_fast_answer_not_zeroed_in_fast_budget():
    model = _make_perf_model()

    assert model.reward(3.0, 5) == 1.0


def test_perf_fast_answer_in_balanced_budget_follows_curve():
    model = _make_perf_model()

    min_realistic, target = model._thresholds_for(15)
    assert min_realistic == 1.0
    assert target == 9.0

    assert model.reward(3.0, 15) == 1.0
    assert model.reward(0.9, 15) == 0.0


def test_perf_decay_within_budget():
    model = _make_perf_model()

    # within budget (target 9 .. budget 15): plateau decays 1.0 -> 0.5
    reward = model.reward(12.0, 15)
    assert 0.0 < reward < 1.0
    # graceful overage band (budget 15 .. 15 + 0.5*15): slightly-late still scores
    assert 0.0 < model.reward(16.0, 15) < 0.5
    # far past the overage band -> zero
    assert model.reward(23.0, 15) == 0.0


def test_perf_falls_back_to_fixed_when_budget_missing():
    model = _make_perf_model()

    assert model._thresholds_for(0) == (1.0, 3.0)
    assert model.reward(0.5, 0) == 0.0
    assert model.reward(3.0, 0) == 1.0


def test_perf_floor_is_mode_dependent():
    from types import SimpleNamespace

    from desearch.protocol import SearchMode
    from neurons.validators.reward.performance_reward import perf_floor_for

    def resp(mode):
        return SimpleNamespace(mode=mode)

    assert perf_floor_for(resp(SearchMode.FAST), 0.50) == 0.40
    assert perf_floor_for(resp(SearchMode.BALANCED), 0.50) == 0.50
    assert perf_floor_for(resp(SearchMode.DEEP), 0.50) == 0.85
    assert perf_floor_for(resp(None), 0.70) == 0.70


def _penalty_model():
    return MinRealisticTimePenaltyModel()


def _resp(process_time, budget):
    return SimpleNamespace(
        dendrite=SimpleNamespace(process_time=process_time),
        max_execution_time=budget,
    )


def test_penalty_not_applied_to_fast_index_answer():
    model = _penalty_model()

    assert model.penalty_for(_resp(3.0, 5)) == 0.0


def test_penalty_applied_below_mode_threshold():
    model = _penalty_model()

    assert model.penalty_for(_resp(0.9, 5)) == model.max_penalty


def test_penalty_falls_back_when_budget_missing():
    model = _penalty_model()

    assert model.penalty_for(_resp(0.5, None)) == model.max_penalty
    assert model.penalty_for(_resp(3.0, None)) == 0.0
