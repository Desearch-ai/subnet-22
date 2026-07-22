import json

from app.domains.miners.router import _parse_windows
from app.domains.miners.schemas import MinerDetail, MinerTypeState

OLD_AI = {
    "verified": 47,
    "declared": 65,
    "quality_avg": 0.85,
    "unreachable_since": None,
}
NEW_AI = {
    "modes": {
        "fast": {"verified": 31, "declared": 100, "quality_avg": 0.62},
        "balanced": {"verified": 31, "declared": 100, "quality_avg": 0.62},
        "deep": {"verified": 31, "declared": 100, "quality_avg": 0.62},
    }
}
WINDOW = {
    "window_start": "2026-07-22T10:00:00+00:00",
    "quality_score": 0.8,
    "passed": True,
    "verified_concurrency": 30,
}


def test_accepts_pre_lane_validator_state():
    assert MinerTypeState(**OLD_AI).verified == 47


def test_accepts_per_mode_validator_state_and_keeps_modes():
    state = MinerTypeState(**NEW_AI)
    assert state.modes["fast"].verified == 31
    assert json.loads(state.model_dump_json())["modes"]["deep"]["quality_avg"] == 0.62


def test_windows_accept_list_and_per_mode_dict():
    detail = MinerDetail(
        hotkey="h",
        uid=1,
        coldkey="c",
        per_type={"ai_search": MinerTypeState(**NEW_AI)},
        windows={
            st: _parse_windows(ws)
            for st, ws in {
                "ai_search": {"fast": [WINDOW], "balanced": [], "deep": [WINDOW]},
                "x_search": [WINDOW],
            }.items()
        },
    )
    out = json.loads(detail.model_dump_json())
    assert len(out["windows"]["ai_search"]["fast"]) == 1
    assert len(out["windows"]["x_search"]) == 1
