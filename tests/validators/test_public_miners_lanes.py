import neurons.validators.api as api
from desearch.miner_config import SearchType, lane_key
from desearch.protocol import SearchMode


def _row(
    uid, key, verified, declared, quality, hotkey="hk", coldkey="ck", unreachable=None
):
    return {
        "uid": uid,
        "hotkey": hotkey,
        "coldkey": coldkey,
        "search_type": key,
        "verified": verified,
        "declared": declared,
        "quality_avg": quality,
        "unreachable_since": unreachable,
    }


FAST = lane_key((SearchType.AI_SEARCH, SearchMode.FAST))
BAL = lane_key((SearchType.AI_SEARCH, SearchMode.BALANCED))
DEEP = lane_key((SearchType.AI_SEARCH, SearchMode.DEEP))
X = lane_key((SearchType.X_SEARCH, None))


def test_per_type_nests_modes_under_ai_search():
    rows = [
        _row(1, FAST, 60, 100, 0.90),
        _row(1, BAL, 20, 100, 0.60),
        _row(1, DEEP, 20, 100, 0.50),
        _row(1, X, 25, 80, 0.80),
    ]
    per_type = api._per_type_from_rows(rows)

    assert set(per_type) == {"ai_search", "x_search"}
    assert set(per_type["ai_search"]["modes"]) == {"fast", "balanced", "deep"}
    assert per_type["ai_search"]["modes"]["fast"]["verified"] == 60
    assert "modes" not in per_type["x_search"]


def test_ai_aggregate_sums_modes_and_weights_quality():
    rows = [
        _row(1, FAST, 60, 100, 0.90),
        _row(1, BAL, 20, 100, 0.60),
        _row(1, DEEP, 20, 100, 0.50),
    ]
    ai = api._per_type_from_rows(rows)["ai_search"]
    assert ai["verified"] == 100
    assert ai["declared"] == 300
    assert ai["quality_avg"] == (0.90 * 60 + 0.60 * 20 + 0.50 * 20) / 100


def test_retired_search_types_are_dropped():
    rows = [_row(1, FAST, 10, 100, 0.7), _row(1, "web_search", 5, 6, 0.2)]
    assert "web_search" not in api._per_type_from_rows(rows)


def test_missing_mode_falls_back_to_empty_state():
    per_type = api._per_type_from_rows([_row(1, FAST, 10, 100, 0.7)])
    assert per_type["ai_search"]["modes"]["deep"]["declared"] == 0
    assert per_type["ai_search"]["declared"] == 100


def test_ai_window_aggregate_is_verified_weighted():
    ai_windows = [
        [
            {
                "window_start": "T1",
                "quality_score": 0.9,
                "passed": True,
                "verified_concurrency": 60,
            }
        ],
        [
            {
                "window_start": "T1",
                "quality_score": 0.6,
                "passed": True,
                "verified_concurrency": 20,
            }
        ],
        [
            {
                "window_start": "T1",
                "quality_score": 0.5,
                "passed": False,
                "verified_concurrency": 20,
            }
        ],
    ]
    agg = api._aggregate_ai_windows(ai_windows)
    assert len(agg) == 1
    w = agg[0]
    assert w["verified_concurrency"] == 100
    assert w["quality_score"] == (0.9 * 60 + 0.6 * 20 + 0.5 * 20) / 100
    assert w["passed"] is False
