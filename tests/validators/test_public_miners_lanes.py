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


def test_ai_search_carries_only_modes():
    rows = [
        _row(1, FAST, 60, 100, 0.90),
        _row(1, BAL, 20, 100, 0.60),
        _row(1, DEEP, 20, 100, 0.50),
        _row(1, X, 25, 80, 0.80),
    ]
    per_type = api._per_type_from_rows(rows)

    assert set(per_type) == {"ai_search", "x_search"}
    assert per_type["ai_search"] == {
        "modes": {
            "fast": {
                "verified": 60,
                "declared": 100,
                "quality_avg": 0.90,
                "unreachable_since": None,
            },
            "balanced": {
                "verified": 20,
                "declared": 100,
                "quality_avg": 0.60,
                "unreachable_since": None,
            },
            "deep": {
                "verified": 20,
                "declared": 100,
                "quality_avg": 0.50,
                "unreachable_since": None,
            },
        }
    }
    assert per_type["x_search"]["verified"] == 25


def test_retired_search_types_are_dropped():
    rows = [_row(1, FAST, 10, 100, 0.7), _row(1, "web_search", 5, 6, 0.2)]
    assert "web_search" not in api._per_type_from_rows(rows)


def test_missing_mode_falls_back_to_empty_state():
    per_type = api._per_type_from_rows([_row(1, FAST, 10, 100, 0.7)])
    modes = per_type["ai_search"]["modes"]
    assert modes["deep"]["declared"] == 0
    assert modes["deep"]["verified"] == 1
    assert modes["fast"]["declared"] == 100


def test_response_model_serializes_modes_without_aggregate():
    rows = [
        _row(1, FAST, 60, 100, 0.90),
        _row(1, BAL, 20, 100, 0.60),
        _row(1, DEEP, 20, 100, 0.50),
        _row(1, X, 25, 80, 0.80),
    ]
    item = api.MinerListItemOut(
        hotkey="hk", uid=1, coldkey="ck", per_type=api._per_type_from_rows(rows)
    )
    dumped = item.model_dump(exclude_none=True)

    assert set(dumped["per_type"]["ai_search"]) == {"modes"}
    assert dumped["per_type"]["ai_search"]["modes"]["fast"]["verified"] == 60
    assert "modes" not in dumped["per_type"]["x_search"]
