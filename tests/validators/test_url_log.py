from neurons.validators.scoring import (
    SNIPPET_CHARS,
    WINDOW_CHARS,
    first_difference,
    url_log,
)

LONG = "the quick brown fox " * 80


def row(url: str, text: str = "", error: str | None = None, status: int = 200) -> dict:
    return {"url": url, "text": text, "error": error, "status": status}


def test_every_row_is_listed_and_only_samples_carry_text():
    rows = [row("https://a/1", LONG), row("https://a/2", error="http_4xx", status=404)]
    sample = {"url": "https://a/1", "outcome": "matched", "similarity": 1.0}

    first, second = url_log(rows, [sample], {"https://a/1": LONG})

    assert first["sampled"] and first["outcome"] == "matched"
    assert len(first["miner_snippet"]) == SNIPPET_CHARS and first["diff_at"] is None
    assert second == {
        "url": "https://a/2",
        "status": 404,
        "error": "http_4xx",
        "text_chars": 0,
        "sampled": False,
        "rejected": False,
    }


def test_a_mismatch_shows_the_text_around_where_it_diverges():
    theirs = LONG[:1000] + "a different ending" + LONG[1000:]
    sample = {"url": "https://a/1", "outcome": "mismatched"}

    (detail,) = url_log([row("https://a/1", LONG)], [sample], {"https://a/1": theirs})

    assert detail["diff_at"] == first_difference(LONG, theirs) == 1000
    assert (
        len(detail["miner_window"])
        == len(detail["validator_window"])
        == 2 * WINDOW_CHARS
    )
    assert "a different ending" in detail["validator_window"]


def test_a_sample_the_validator_could_not_read_has_no_divergence():
    sample = {"url": "https://a/1", "outcome": "not_fetched"}

    (detail,) = url_log([row("https://a/1", LONG)], [sample], {})

    assert detail["validator_snippet"] == "" and detail["diff_at"] is None


def test_the_task_api_accepts_every_field_the_validator_logs(monkeypatch):
    from pathlib import Path

    from neurons.validators.scoring import FetchedPage, judge_sample

    monkeypatch.syspath_prepend(str(Path(__file__).parents[2] / "task-api"))
    from app.models import UrlDetail

    page = {**row("https://a/1", LONG), "title": "", "final_url": "https://a/1"}
    sample = judge_sample(page, FetchedPage(200, f"<p>{LONG}</p>", via="own_ip"))
    rows = [page, row("https://a/2", error="timeout", status=0)]

    for detail in url_log(rows, [sample], {"https://a/1": LONG}):
        assert UrlDetail.model_validate(detail).model_dump(exclude_unset=True) == detail
