import pytest

from neurons.validators.scoring.constants import GATE_RAMP, QUALITY_THRESHOLDS
from neurons.validators.scoring.constants import MIN_DEEP_SAMPLES_PER_POOL
from neurons.validators.scoring.query_scheduler import combine_pool_scores

AI_FAST = ("ai_search", "fast")


REFERENCE = (0.95, 0.95, 50, MIN_DEEP_SAMPLES_PER_POOL)


def _pay(q_gate, q_weight=0.80, volume=50, pool_key=AI_FAST):
    pool = {
        0: (q_gate, q_weight, volume, MIN_DEEP_SAMPLES_PER_POOL),
        1: REFERENCE,
    }
    return combine_pool_scores({pool_key: pool}).get(0, 0.0)


def test_above_threshold_pay_is_set_by_weight_not_gate():
    thr = QUALITY_THRESHOLDS["ai_search"]
    assert _pay(thr) > 0
    assert _pay(thr) == pytest.approx(_pay(thr + 0.20))


def test_higher_gate_wins_more_of_the_pool():
    thr = QUALITY_THRESHOLDS["ai_search"]
    assert _pay(thr) > _pay(thr - GATE_RAMP * 0.2) > _pay(thr - GATE_RAMP / 2)


def test_partial_gate_costs_share_of_the_pool():
    thr = QUALITY_THRESHOLDS["ai_search"]
    full = _pay(thr)
    assert 0 < _pay(thr - GATE_RAMP / 2) < full
    assert 0 < _pay(thr - GATE_RAMP * 0.2) < full


def test_ramp_gives_partial_credit_just_below_threshold():
    thr = QUALITY_THRESHOLDS["ai_search"]
    assert 0 < _pay(thr - GATE_RAMP / 2) < _pay(thr)


def test_exact_zero_at_band_bottom():
    thr = QUALITY_THRESHOLDS["ai_search"]
    assert _pay(thr - GATE_RAMP) == 0.0


def test_below_ramp_band_is_zero():
    thr = QUALITY_THRESHOLDS["ai_search"]
    assert _pay(thr - GATE_RAMP - 0.01) == 0.0


def test_zero_volume_serves_nothing():
    assert _pay(0.90, volume=0) == 0.0


def test_x_search_threshold_ramp():
    thr = QUALITY_THRESHOLDS["x_search"]

    def pay_x(q):
        return _pay(q, pool_key=("x_search", None))

    assert pay_x(thr) > 0
    assert pay_x(thr - GATE_RAMP - 0.01) == 0.0
    assert 0 < pay_x(thr - GATE_RAMP / 2) < pay_x(thr)


def test_gate_and_volume_credit_compose():
    thr = QUALITY_THRESHOLDS["ai_search"]
    pool = {
        0: (0.90, 0.80, 100, MIN_DEEP_SAMPLES_PER_POOL),
        1: (thr - GATE_RAMP / 2, 0.80, 50, MIN_DEEP_SAMPLES_PER_POOL),
    }
    result = combine_pool_scores({AI_FAST: pool})
    assert result[0] > 0
    assert 0 < result.get(1, 0.0) < result[0]


if __name__ == "__main__":
    test_above_threshold_pay_is_set_by_weight_not_gate()
    test_partial_gate_costs_share_of_the_pool()
    test_higher_gate_wins_more_of_the_pool()
    test_ramp_gives_partial_credit_just_below_threshold()
    test_exact_zero_at_band_bottom()
    test_below_ramp_band_is_zero()
    test_zero_volume_serves_nothing()
    test_x_search_threshold_ramp()
    test_gate_and_volume_credit_compose()
    print("ok")
