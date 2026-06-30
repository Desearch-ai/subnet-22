import json
from pathlib import Path

from desearch.dataset.hf_dataset import HFQuestionPool


X_FIXTURE = [
    {
        "id": "x1",
        "question": "What did the announcement say?",
        "difficulty": "medium",
        "start_date": "2026-06-01T00:00:00Z",
        "end_date": "2026-06-02T00:00:00Z",
    },
    {
        "id": "x2",
        "question": "Who broke the news first?",
        "difficulty": "hard",
        "start_date": "2026-06-03T00:00:00Z",
        "end_date": "2026-06-04T00:00:00Z",
    },
]

NEWS_FIXTURE = [
    {"id": "n1", "question": "What happened in the market today?"},
    {"id": "n2", "question": "Summarize the latest policy change."},
]


def _write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(r) for r in rows))


def _make_pool(tmp_path):
    cache = tmp_path / "cache"
    x_file = cache / "x" / "x-2026-06-01.jsonl"
    news_file = cache / "questions" / "q-2026-06-01.jsonl"
    _write_jsonl(x_file, X_FIXTURE)
    _write_jsonl(news_file, NEWS_FIXTURE)

    pool = HFQuestionPool(cache_dir=str(cache))
    pool._sync_from_hf = lambda: [x_file, news_file]
    pool._load_datasets = lambda: []
    return pool


def test_x_rows_tagged_and_dates_preserved(tmp_path):
    pool = _make_pool(tmp_path)
    rows = pool._load_rows()

    x_rows = [r for r in rows if r["lane"] == "x"]
    assert len(x_rows) == 2
    for r in x_rows:
        assert r["start_date"] is not None
        assert r["end_date"] is not None
    assert {r["id"] for r in x_rows} == {"x1", "x2"}
    assert x_rows[0]["start_date"] == "2026-06-01T00:00:00Z"


def test_sample_lane_returns_only_that_lane(tmp_path):
    pool = _make_pool(tmp_path)

    out = pool.sample_lane("x", 5)
    assert out is not None
    assert all(r["lane"] == "x" for r in out)
    assert len(out) == 2

    capped = pool.sample_lane("x", 1)
    assert len(capped) == 1
    assert capped[0]["lane"] == "x"

    assert pool.sample_lane("does-not-exist", 3) is None


def test_news_and_x_lanes_coexist(tmp_path):
    pool = _make_pool(tmp_path)
    rows = pool._ensure_rows()

    lanes = {r["lane"] for r in rows}
    assert "x" in lanes
    assert "news" in lanes

    news_rows = [r for r in rows if r["lane"] == "news"]
    assert all(r["start_date"] is None for r in news_rows)


if __name__ == "__main__":
    import tempfile

    for fn in (
        test_x_rows_tagged_and_dates_preserved,
        test_sample_lane_returns_only_that_lane,
        test_news_and_x_lanes_coexist,
    ):
        with tempfile.TemporaryDirectory() as d:
            fn(Path(d))
        print(f"PASS {fn.__name__}")
    print("ALL PASS")
