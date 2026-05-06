from neurons.validators.scoring.capacity import (
    DEFAULT_PER_UID,
    HARD_CAP_PER_UID,
    QUALITY_THRESHOLD,
    RAMP_STEP_PCT,
    SYNTHETIC_BUDGET_PER_TYPE,
    allocate_synthetic_budget,
)


def _all_declared(uids, value=10**6):
    return {uid: value for uid in uids}


def _unconstrained_prev(uids, value=10**6):
    """Use a huge prev allocation so ramp_cap never binds — for tests that
    aren't exercising the ramp."""
    return {uid: value for uid in uids}


def _allocate(quality, declared=None, prev=None):
    declared = declared if declared is not None else _all_declared(quality)
    prev = prev if prev is not None else _unconstrained_prev(quality)
    return allocate_synthetic_budget(quality, declared, prev)


def test_empty_input_returns_empty():
    assert allocate_synthetic_budget({}, {}, {}) == {}


def test_all_zero_quality_falls_back_to_floor():
    quality = {1: 0.0, 2: 0.0, 3: 0.0}
    out = _allocate(quality)
    assert out == {1: DEFAULT_PER_UID, 2: DEFAULT_PER_UID, 3: DEFAULT_PER_UID}


def test_single_high_quality_capped_at_hard_cap():
    """One eligible miner can't soak the whole budget — HARD_CAP_PER_UID
    bounds any single UID's allocation."""
    out = _allocate({1: 0.9})
    assert out[1] == HARD_CAP_PER_UID


def test_declared_caps_allocation():
    quality = {1: 0.9, 2: 0.5}
    declared = {1: 5, 2: 10**6}
    out = _allocate(quality, declared=declared)
    assert out[1] == 5
    assert out[2] >= DEFAULT_PER_UID


def test_total_bounded_by_floor_plus_budget():
    quality = {uid: 0.5 + 0.01 * uid for uid in range(60)}
    out = _allocate(quality)
    assert sum(out.values()) <= len(quality) * DEFAULT_PER_UID + SYNTHETIC_BUDGET_PER_TYPE + 1


def test_clones_split_team_share():
    """Two equal-quality clones each get half the share of one solo UID
    at the same quality (when others are present)."""
    others = {uid: 0.5 for uid in range(2, 12)}

    solo = {1: 0.9, **others}
    split = {1: 0.9, 100: 0.9, **others}

    solo_out = _allocate(solo)
    split_out = _allocate(split)

    split_share = (split_out[1] - DEFAULT_PER_UID) + (split_out[100] - DEFAULT_PER_UID)

    assert split_out[1] < solo_out[1]
    assert split_out[100] < solo_out[1]
    assert split_share <= SYNTHETIC_BUDGET_PER_TYPE + 1


def test_higher_quality_gets_more_share():
    quality = {1: 0.9, 2: 0.5, 3: QUALITY_THRESHOLD + 0.01}
    out = _allocate(quality)
    assert out[1] > out[2] > out[3]


def test_below_threshold_uids_get_floor_only():
    """A miner whose quality EMA is below the threshold is excluded from the
    share split — only the floor."""
    quality = {1: 0.9, 2: QUALITY_THRESHOLD - 0.01}
    out = _allocate(quality)
    assert out[2] == DEFAULT_PER_UID
    assert out[1] == HARD_CAP_PER_UID


def test_threshold_exactly_qualifies():
    quality = {1: QUALITY_THRESHOLD, 2: QUALITY_THRESHOLD - 0.01}
    out = _allocate(quality)
    assert out[1] == HARD_CAP_PER_UID
    assert out[2] == DEFAULT_PER_UID


def test_hard_cap_binds_below_declared():
    """HARD_CAP_PER_UID applies even when the miner declares more — a single
    UID can't consume more than the cap regardless of quality or declared."""
    out = _allocate({1: 0.95}, declared={1: 500})
    assert out[1] == HARD_CAP_PER_UID


def test_declared_below_hard_cap_still_caps():
    """When declared < HARD_CAP, declared still binds."""
    out = _allocate({1: 0.95}, declared={1: 25})
    assert out[1] == 25


def test_ramp_cap_bounds_growth_from_floor():
    """A new miner with high quality can't jump straight to a big allocation;
    ramp_cap = prev + 10% of declared bounds the upward step."""
    out = _allocate({1: 0.9}, declared={1: 100}, prev={1: 1})
    # ramp_cap = 1 + 100*0.10 = 11; share would award much more.
    assert out[1] == 11


def test_ramp_cap_grows_each_epoch():
    """Repeated allocation walks: floor -> +10/epoch as the miner's prev grows."""
    declared = {1: 100}
    quality = {1: 0.9}
    expected = [11, 21, 31, 41, 51, 61, 71, 81, 91, 100]  # cap at HARD_CAP=100
    prev = 1
    for step in expected:
        out = _allocate(quality, declared=declared, prev={1: prev})
        assert out[1] == step
        prev = out[1]


def test_ramp_cap_doesnt_constrain_when_share_is_smaller():
    """If the quality-share would be smaller than the ramp ceiling, share
    wins — the ramp cap is one-way (only constrains growth)."""
    quality = {uid: 0.9 for uid in range(1, 11)}
    quality[100] = QUALITY_THRESHOLD + 0.01
    out = _allocate(quality, prev={uid: 80 for uid in quality})
    # uid 100's share is small in the sea of high-q miners; with prev=80,
    # ramp_cap would be 90, but share binds well below that.
    assert out[100] < 11


def test_quality_drop_decays_via_share_not_ramp():
    """Allocation can drop more than ramp_step in one epoch when quality
    drops — ramp_cap only caps growth, not decay."""
    declared = {1: 100, 2: 100}
    out = _allocate(
        {1: QUALITY_THRESHOLD + 0.01, 2: 0.9},
        declared=declared,
        prev={1: 60, 2: 60},
    )
    # uid 1's share is now small; final allocation < 60 even though prev was 60.
    assert out[1] < 60


def test_below_threshold_resets_to_floor_regardless_of_prev():
    """A miner whose EMA dips below threshold drops straight to floor — no
    smoothed multiplicative decay of prev allocation."""
    out = _allocate(
        {1: QUALITY_THRESHOLD - 0.01},
        declared={1: 100},
        prev={1: 50},
    )
    assert out[1] == DEFAULT_PER_UID


def test_ramp_step_minimum_one_for_small_declared():
    """Even if declared * RAMP_STEP_PCT rounds to zero, ramp grows by at least
    1 per epoch so small-declared miners aren't stuck."""
    # 5 * 0.10 = 0.5 → int() = 0 → max(1, 0) = 1
    assert int(5 * RAMP_STEP_PCT) == 0
    out = _allocate({1: 0.9}, declared={1: 5}, prev={1: 1})
    # ramp_cap = 1 + max(1, 0) = 2; declared=5; share huge → ceiling = 2.
    assert out[1] == 2
