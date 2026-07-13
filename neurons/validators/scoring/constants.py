SEARCH_TYPE_WEIGHTS = {
    "ai_search": 0.90,
    "x_search": 0.10,
}

QUALITY_THRESHOLDS: dict[str, float] = {
    "ai_search": 0.50,
    "x_search": 0.60,
}

QUALITY_EXPONENT = 3.0
VOLUME_EXPONENT = 2.0
COVERAGE_EXPONENT = 2.0
MIN_VOLUME_RATIO = 0.70
GATE_RAMP = 0.05

DEFAULT_PER_UID = 1
