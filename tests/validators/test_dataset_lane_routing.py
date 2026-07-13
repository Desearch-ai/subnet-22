from desearch.protocol import ResultType
from neurons.validators.scoring.synthetic_query_generator import (
    SyntheticQueryGenerator,
    TWITTER_TOOL,
)


X_ROWS = [
    {
        "id": f"x{i}",
        "question": f"x lane question {i}",
        "start_date": f"2026-06-1{i}T00:00:00Z",
        "end_date": f"2026-06-2{i}T00:00:00Z",
        "lane": "x",
    }
    for i in range(5)
]

WEB_ROWS = {
    "news": [
        {
            "id": f"n{i}",
            "question": f"news lane question {i}",
            "start_date": None,
            "end_date": None,
            "lane": "news",
        }
        for i in range(5)
    ],
    "squad": [
        {
            "id": f"s{i}",
            "question": f"squad lane question {i}",
            "start_date": None,
            "end_date": None,
            "lane": "squad",
        }
        for i in range(5)
    ],
    "nq": [
        {
            "id": f"q{i}",
            "question": f"nq lane question {i}",
            "start_date": None,
            "end_date": None,
            "lane": "nq",
        }
        for i in range(5)
    ],
}

X_QUESTIONS = {r["question"] for r in X_ROWS}
X_BY_QUESTION = {r["question"]: r for r in X_ROWS}


class FakePool:
    def __init__(self, x_rows=X_ROWS, web_rows=WEB_ROWS):
        self.x_rows = x_rows
        self.web_rows = web_rows

    def sample_lane(self, lane, n):
        import random

        if lane == "x":
            rows = list(self.x_rows)
        else:
            rows = list(self.web_rows.get(lane, []))
        if not rows:
            return None
        random.shuffle(rows)
        return rows[:n]


class FakeBasic:
    def generate_random_x_query(self):
        return "FALLBACK_X_QUERY"


def _make_gen(pool):
    gen = object.__new__(SyntheticQueryGenerator)
    gen.hf_pool = pool
    gen.basic_dataset = FakeBasic()
    return gen


def _ordering(items):
    return [i["query"]["query"] for i in items]


def test_x_search_uses_basic_not_dataset():
    # Basic x_search is unchanged: it uses the basic random X query, NOT the X
    # dataset. The X dataset drives the AI-search Twitter tool (see below).
    gen = _make_gen(FakePool())
    uids = [1, 2, 3]

    items = gen._generate_dataset_queries(uids, {})

    x_items = [i for i in items if i["search_type"] == "x_search"]
    assert x_items
    for it in x_items:
        q = it["query"]
        assert q["query"] == "FALLBACK_X_QUERY"
        assert "start_date" not in q and "end_date" not in q


def test_ai_twitter_tool_prefers_x_lane():
    gen = _make_gen(FakePool())

    items = gen._generate_dataset_queries(list(range(8)), {})

    ai_x = [
        i
        for i in items
        if i["search_type"] == "ai_search" and TWITTER_TOOL in i["query"]["tools"]
    ]
    assert ai_x, "expected at least one ai_search routed to the Twitter tool"
    for it in ai_x:
        assert it["query"]["query"] in X_QUESTIONS
        # dataset queries are mode-free; ai+Twitter carries the X date window
        assert it["query"]["start_date"] and it["query"]["end_date"]


def test_x_fallback_when_no_x_lane():
    pool = FakePool(x_rows=[])
    gen = _make_gen(pool)

    items = gen._generate_dataset_queries([1, 2], {})

    x_items = [i for i in items if i["search_type"] == "x_search"]
    assert x_items
    for it in x_items:
        assert it["query"]["query"] == "FALLBACK_X_QUERY"


def test_returns_none_when_pool_empty():
    pool = FakePool(x_rows=[], web_rows={"news": [], "squad": [], "nq": []})
    gen = _make_gen(pool)

    assert gen._generate_dataset_queries([1, 2], {}) is None


def test_ai_combos_balance_modes_and_tools_in_a_window_of_six():
    from neurons.validators.scoring.synthetic_query_generator import (
        _ai_combos,
        WEB_TOOL,
        TWITTER_TOOL,
    )

    for _ in range(100):
        combos = _ai_combos(6)
        assert len({c[0] for c in combos}) == 3
        assert {c[1][0] for c in combos} == {WEB_TOOL, TWITTER_TOOL}


def test_ai_combos_full_cycle_keep_4to1_result_type_ratio():
    from collections import Counter
    from neurons.validators.scoring.synthetic_query_generator import _ai_combos

    for _ in range(20):
        counts = Counter(c[2] for c in _ai_combos(30))
        assert counts[ResultType.LINKS_WITH_FINAL_SUMMARY.value] == 24
        assert counts[ResultType.ONLY_LINKS.value] == 6


def test_ai_combos_empty_for_non_positive():
    from neurons.validators.scoring.synthetic_query_generator import _ai_combos

    assert _ai_combos(0) == []


def test_weighted_counts_always_sums_to_n():
    from neurons.validators.scoring.synthetic_query_generator import _weighted_counts

    weight_sets = (
        [0.6, 0.2, 0.2],
        [0.5, 0.5],
        [1.0, 0.0],
        [0.25, 0.25, 0.25, 0.25],
        [0.7, 0.3],
    )
    for n in (0, 1, 2, 5, 10, 37):
        for weights in weight_sets:
            counts = _weighted_counts(n, weights)
            assert sum(counts) == n
            assert all(c >= 0 for c in counts)


def test_ai_combos_mode_split_is_60_20_20():
    from collections import Counter

    from neurons.validators.scoring.synthetic_query_generator import _ai_combos

    for _ in range(50):
        modes = Counter(getattr(m, "value", m) for m, _, _ in _ai_combos(5))
        assert (modes["fast"], modes["balanced"], modes["deep"]) == (3, 1, 1)

    modes = Counter(getattr(m, "value", m) for m, _, _ in _ai_combos(10))
    assert (modes["fast"], modes["balanced"], modes["deep"]) == (6, 2, 2)


def test_ai_combos_mode_and_result_type_are_independent():
    from neurons.validators.scoring.synthetic_query_generator import _ai_combos

    deep_total = deep_and_links = 0
    for _ in range(3000):
        for m, _, rt in _ai_combos(10):
            if getattr(m, "value", m) == "deep":
                deep_total += 1
                if rt == ResultType.ONLY_LINKS.value:
                    deep_and_links += 1

    p_links_given_deep = deep_and_links / deep_total
    assert 0.12 < p_links_given_deep < 0.28


def test_single_uid_gets_balanced_coverage_across_its_queries():
    gen = _make_gen(FakePool())

    items = gen._generate_dataset_queries([1], {"ai_search": {1: 6}})
    ai = [i for i in items if i["search_type"] == "ai_search"]

    assert len(ai) == 6
    assert len({i["query"]["mode"] for i in ai}) == 3
    assert len({i["query"]["tools"][0] for i in ai}) == 2


def test_ai_search_randomizes_result_type():
    gen = _make_gen(FakePool())
    valid_values = {rt.value for rt in ResultType}

    seen = set()
    for _ in range(5):
        items = gen._generate_dataset_queries(list(range(40)), {})
        ai_items = [i for i in items if i["search_type"] == "ai_search"]
        assert ai_items
        for it in ai_items:
            assert it["query"]["result_type"] in valid_values
            seen.add(it["query"]["result_type"])
        for it in items:
            if it["search_type"] != "ai_search":
                assert "result_type" not in it["query"]

    assert seen == valid_values


def test_randomness_across_invocations():
    gen = _make_gen(FakePool())
    uids = list(range(12))

    first = _ordering(gen._generate_dataset_queries(uids, {}))
    seen_diff = False
    for _ in range(10):
        nxt = _ordering(gen._generate_dataset_queries(uids, {}))
        if nxt != first:
            seen_diff = True
            break
    assert seen_diff, "expected different question ordering across epochs"


if __name__ == "__main__":
    test_x_search_uses_basic_not_dataset()
    test_ai_twitter_tool_prefers_x_lane()
    test_x_fallback_when_no_x_lane()
    test_returns_none_when_pool_empty()
    test_ai_search_randomizes_result_type()
    test_ai_combos_balance_modes_and_tools_in_a_window_of_six()
    test_ai_combos_full_cycle_keep_4to1_result_type_ratio()
    test_ai_combos_empty_for_non_positive()
    test_ai_combos_mode_split_is_60_20_20()
    test_ai_combos_mode_and_result_type_are_independent()
    test_single_uid_gets_balanced_coverage_across_its_queries()
    test_randomness_across_invocations()
    print("ok")
