import pytest

from neurons.validators.scoring.constants import GATE_RAMP, QUALITY_THRESHOLDS
from neurons.validators.scoring.query_scheduler import combine_superlinear_scores


def _pay(q_gate, q_weight=0.80, volume=50):
    return combine_superlinear_scores(
        {"ai_search": {0: (q_gate, q_weight, volume)}}
    ).get(0, 0.0)


def test_above_threshold_pay_is_set_by_weight_not_gate():
    thr = QUALITY_THRESHOLDS["ai_search"]
    assert _pay(thr) > 0
    assert _pay(thr) == pytest.approx(_pay(thr + 0.20))


def test_exact_gate_value_via_cube():
    thr = QUALITY_THRESHOLDS["ai_search"]
    full = _pay(thr)
    assert _pay(thr - GATE_RAMP / 2) == pytest.approx(full * 0.5**3, rel=1e-6)
    assert _pay(thr - GATE_RAMP * 0.2) == pytest.approx(full * 0.8**3, rel=1e-6)


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
        return combine_superlinear_scores({"x_search": {0: (q, 0.80, 50)}}).get(0, 0.0)

    assert pay_x(thr) > 0
    assert pay_x(thr - GATE_RAMP - 0.01) == 0.0
    assert 0 < pay_x(thr - GATE_RAMP / 2) < pay_x(thr)


def test_gate_and_volume_credit_compose():
    thr = QUALITY_THRESHOLDS["ai_search"]
    result = combine_superlinear_scores(
        {"ai_search": {0: (0.90, 0.80, 100), 1: (thr - GATE_RAMP / 2, 0.80, 50)}}
    )
    assert result[0] > 0
    assert 0 < result.get(1, 0.0) < result[0]


if __name__ == "__main__":
    test_above_threshold_pay_is_set_by_weight_not_gate()
    test_exact_gate_value_via_cube()
    test_ramp_gives_partial_credit_just_below_threshold()
    test_exact_zero_at_band_bottom()
    test_below_ramp_band_is_zero()
    test_zero_volume_serves_nothing()
    test_x_search_threshold_ramp()
    test_gate_and_volume_credit_compose()
    print("ok")
