from neurons.validators.scoring.capacity import (
    DEFAULT_PER_UID,
    HARD_CAP_PER_UID,
    QUALITY_THRESHOLD,
    RAMP_FRACTION,
    next_verified,
)


def test_ramp_up_when_quality_passes():
    """At quality >= threshold, verified grows by max(1, declared * RAMP_FRACTION)."""
    assert next_verified(current=1, declared=100, quality_avg=0.9) == 11
    assert next_verified(current=50, declared=100, quality_avg=0.5) == 60


def test_ramp_down_when_quality_fails():
    """Below threshold, verified decays by the same step."""
    assert next_verified(current=50, declared=100, quality_avg=0.1) == 40


def test_decay_floors_at_default():
    assert next_verified(current=5, declared=100, quality_avg=0.0) == DEFAULT_PER_UID


def test_ramp_caps_at_declared():
    assert next_verified(current=95, declared=100, quality_avg=0.9) == 100
    assert next_verified(current=24, declared=25, quality_avg=0.9) == 25


def test_ramp_caps_at_hard_cap():
    assert next_verified(current=95, declared=500, quality_avg=0.9) == HARD_CAP_PER_UID


def test_ramp_step_minimum_one_for_small_declared():
    """5 * 0.10 = 0.5, rounds to 0 — but ramp still moves by 1 per epoch."""
    assert int(5 * RAMP_FRACTION) == 0
    assert next_verified(current=1, declared=5, quality_avg=0.9) == 2
    assert next_verified(current=3, declared=5, quality_avg=0.0) == 2


def test_threshold_exactly_qualifies():
    assert next_verified(current=10, declared=100, quality_avg=QUALITY_THRESHOLD) == 20
    assert (
        next_verified(current=10, declared=100, quality_avg=QUALITY_THRESHOLD - 0.001)
        == 0
        or next_verified(
            current=10, declared=100, quality_avg=QUALITY_THRESHOLD - 0.001
        )
        == DEFAULT_PER_UID
    )


def test_declared_below_default_clamps():
    """Bogus declared=0 is treated as DEFAULT_PER_UID for step purposes."""
    assert next_verified(current=1, declared=0, quality_avg=0.9) == DEFAULT_PER_UID
