from neurons.validators.scoring.capacity import (
    DEFAULT_PER_UID,
    HARD_CAP_PER_UID,
    QUALITY_THRESHOLD,
    SYNTHETIC_BUDGET_PER_TYPE,
    allocate_synthetic_budget,
)


def _all_declared(uids, value=10**6):
    return {uid: value for uid in uids}


def test_empty_input_returns_empty():
    assert allocate_synthetic_budget({}, {}) == {}


def test_all_zero_quality_falls_back_to_floor():
    quality = {1: 0.0, 2: 0.0, 3: 0.0}
    out = allocate_synthetic_budget(quality, _all_declared(quality))
    assert out == {1: DEFAULT_PER_UID, 2: DEFAULT_PER_UID, 3: DEFAULT_PER_UID}


def test_single_high_quality_capped_at_hard_cap():
    """One eligible miner can't soak the whole budget — HARD_CAP_PER_UID
    bounds any single UID's allocation."""
    quality = {1: 0.9}
    out = allocate_synthetic_budget(quality, _all_declared(quality))
    assert out[1] == HARD_CAP_PER_UID


def test_declared_caps_allocation():
    quality = {1: 0.9, 2: 0.5}
    declared = {1: 5, 2: 10**6}
    out = allocate_synthetic_budget(quality, declared)
    assert out[1] == 5
    assert out[2] >= DEFAULT_PER_UID


def test_total_bounded_by_floor_plus_budget():
    quality = {uid: 0.5 + 0.01 * uid for uid in range(60)}
    out = allocate_synthetic_budget(quality, _all_declared(quality))
    assert sum(out.values()) <= len(quality) * DEFAULT_PER_UID + SYNTHETIC_BUDGET_PER_TYPE + 1


def test_clones_split_team_share():
    """Two equal-quality clones each get half the share of one solo UID
    at the same quality (when others are present)."""
    others = {uid: 0.5 for uid in range(2, 12)}

    solo = {1: 0.9, **others}
    split = {1: 0.9, 100: 0.9, **others}

    solo_out = allocate_synthetic_budget(solo, _all_declared(solo))
    split_out = allocate_synthetic_budget(split, _all_declared(split))

    split_share = (split_out[1] - DEFAULT_PER_UID) + (split_out[100] - DEFAULT_PER_UID)

    assert split_out[1] < solo_out[1]
    assert split_out[100] < solo_out[1]
    assert split_share <= SYNTHETIC_BUDGET_PER_TYPE + 1


def test_higher_quality_gets_more_share():
    quality = {1: 0.9, 2: 0.5, 3: 0.4}
    out = allocate_synthetic_budget(quality, _all_declared(quality))
    assert out[1] > out[2] > out[3]


def test_below_threshold_uids_get_floor_only():
    """A miner whose quality EMA is below the threshold is excluded from the
    share split — only the floor."""
    quality = {1: 0.9, 2: QUALITY_THRESHOLD - 0.01}
    out = allocate_synthetic_budget(quality, _all_declared(quality))
    assert out[2] == DEFAULT_PER_UID
    assert out[1] == HARD_CAP_PER_UID


def test_bootstrap_fallback_when_no_one_crosses_threshold():
    """Until any miner has been scored well enough to cross the threshold,
    the budget is shared across all UIDs by their (low) quality."""
    quality = {1: 0.2, 2: 0.1}
    out = allocate_synthetic_budget(quality, _all_declared(quality))
    assert sum(out.values()) > len(quality) * DEFAULT_PER_UID
    assert out[1] > out[2]


def test_threshold_exactly_qualifies():
    quality = {1: QUALITY_THRESHOLD, 2: QUALITY_THRESHOLD - 0.01}
    out = allocate_synthetic_budget(quality, _all_declared(quality))
    assert out[1] == HARD_CAP_PER_UID
    assert out[2] == DEFAULT_PER_UID


def test_hard_cap_binds_below_declared():
    """HARD_CAP_PER_UID applies even when the miner declares more — a single
    UID can't consume more than the cap regardless of quality or declared."""
    quality = {1: 0.95}
    declared = {1: 500}
    out = allocate_synthetic_budget(quality, declared)
    assert out[1] == HARD_CAP_PER_UID


def test_declared_below_hard_cap_still_caps():
    """When declared < HARD_CAP, declared still binds."""
    quality = {1: 0.95}
    declared = {1: 25}
    out = allocate_synthetic_budget(quality, declared)
    assert out[1] == 25
