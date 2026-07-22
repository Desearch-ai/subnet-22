from types import SimpleNamespace

from desearch.protocol import SearchMode

from desearch.protocol import ResultType, SearchMode
from neurons.validators.scoring.query_scheduler import (
    DEEP_SAMPLE_FLOOR,
    QueryScheduler,
)


def _sched():
    return QueryScheduler.__new__(QueryScheduler)


def _resp(
    mode=SearchMode.FAST,
    result_type=ResultType.LINKS_WITH_FINAL_SUMMARY,
    tools=("Web Search",),
):
    return SimpleNamespace(mode=mode, result_type=result_type, tools=list(tools))


def _items(uid, responses):
    return [{"uid": uid, "response": r} for r in responses]


def test_combo_key_reads_response_not_query():
    item = {
        "uid": 1,
        "response": _resp(SearchMode.DEEP, ResultType.ONLY_LINKS, ("Twitter Search",)),
    }
    assert QueryScheduler._deep_combo_key(item) == (
        "deep",
        "ONLY_LINKS",
        ("Twitter Search",),
    )


def test_combo_key_degrades_gracefully_without_response():
    assert QueryScheduler._deep_combo_key({"uid": 1}) == (None, None, ())


def test_deep_floor_is_at_least_three_when_available():
    items = _items(1, [_resp() for _ in range(10)])
    assert len(_sched()._sample_deep_synth(items)) >= DEEP_SAMPLE_FLOOR


def test_deep_scores_all_when_fewer_than_floor():
    items = _items(1, [_resp(), _resp(SearchMode.DEEP)])
    assert len(_sched()._sample_deep_synth(items)) == 2


def test_single_item_uid_gets_one_deep():
    assert len(_sched()._sample_deep_synth(_items(1, [_resp()]))) == 1


def test_deep_sample_is_per_uid():
    items = _items(1, [_resp() for _ in range(10)]) + _items(
        2, [_resp(SearchMode.DEEP) for _ in range(10)]
    )
    picked = _sched()._sample_deep_synth(items)
    for uid in (1, 2):
        assert sum(1 for i in picked if items[i]["uid"] == uid) >= DEEP_SAMPLE_FLOOR


def test_deep_sample_tracks_dispatch_result_type_ratio():
    from collections import Counter

    from neurons.validators.scoring.synthetic_query_generator import _ai_combos

    sched = _sched()
    dispatched = Counter()
    deep = Counter()
    for _ in range(1500):
        combos = _ai_combos(SearchMode.BALANCED, 50)
        items = [
            {
                "uid": 1,
                "response": SimpleNamespace(
                    mode=SearchMode.BALANCED, result_type=rt, tools=t
                ),
            }
            for t, rt in combos
        ]
        for _, rt in combos:
            dispatched[rt] += 1
        for i in sched._sample_deep_synth(items):
            deep[combos[i][1]] += 1

    dispatched_links = dispatched["ONLY_LINKS"] / sum(dispatched.values())
    deep_links = deep["ONLY_LINKS"] / sum(deep.values())
    assert abs(deep_links - dispatched_links) < 0.03


if __name__ == "__main__":
    test_combo_key_reads_response_not_query()
    test_combo_key_degrades_gracefully_without_response()
    test_deep_floor_is_at_least_three_when_available()
    test_deep_scores_all_when_fewer_than_floor()
    test_single_item_uid_gets_one_deep()
    test_deep_sample_is_per_uid()
    test_deep_sample_tracks_dispatch_result_type_ratio()
    print("ok")
