from desearch.miner_config import SearchType
from desearch.protocol import SearchMode

SEARCH_TYPE_WEIGHTS = {
    SearchType.AI_SEARCH: 0.90,
    SearchType.X_SEARCH: 0.10,
}

AI_MODE_WEIGHTS = {
    SearchMode.FAST: 0.60,
    SearchMode.BALANCED: 0.20,
    SearchMode.DEEP: 0.20,
}

POOL_SHARES = {
    (SearchType.AI_SEARCH, mode): SEARCH_TYPE_WEIGHTS[SearchType.AI_SEARCH] * weight
    for mode, weight in AI_MODE_WEIGHTS.items()
}
POOL_SHARES[(SearchType.X_SEARCH, None)] = SEARCH_TYPE_WEIGHTS[SearchType.X_SEARCH]

QUALITY_THRESHOLDS: dict[SearchType, float] = {
    SearchType.AI_SEARCH: 0.50,
    SearchType.X_SEARCH: 0.60,
}

QUALITY_EXPONENT = 3.0
VOLUME_EXPONENT = 2.0
GATE_RAMP = 0.05

MIN_DEEP_SAMPLES_PER_POOL = 3

DEFAULT_PER_UID = 1
