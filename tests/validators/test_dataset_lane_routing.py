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
WEB_QUESTIONS = {r["question"] for lane in WEB_ROWS.values() for r in lane}
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


def test_web_search_sources_web_lanes():
    gen = _make_gen(FakePool())

    items = gen._generate_dataset_queries([1, 2, 3], {})

    web_items = [i for i in items if i["search_type"] == "web_search"]
    assert web_items
    for it in web_items:
        assert it["query"]["query"] in WEB_QUESTIONS


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
    test_web_search_sources_web_lanes()
    test_ai_twitter_tool_prefers_x_lane()
    test_x_fallback_when_no_x_lane()
    test_returns_none_when_pool_empty()
    test_randomness_across_invocations()
    print("ok")
