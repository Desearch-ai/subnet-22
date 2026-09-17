import time

from app.state import connect
from app.validations import URL_DETAIL_DAYS, Validations, build_report

JOB = {"round_id": "r", "miner": "m", "key": "k", "urls": ["https://a.example/"]}
URLS = [{"url": "https://a.example/", "status": 200, "miner_snippet": "Story"}]


def verdict(task_id: str, scored_at: float) -> dict:
    report = build_report(
        task_id, JOB, "v", {"verdict": "pass", "reason": "ok", "urls": URLS}
    )
    return {**report, "scored_at": scored_at}


def test_the_per_url_detail_is_kept_beside_the_verdict_not_in_the_report(tmp_path):
    validations = Validations(connect(str(tmp_path / "db")))
    report = verdict("t1", time.time())
    validations.record(report, URLS)

    assert "urls" not in report
    assert validations.urls("t1") == URLS
    assert validations.urls("missing") == []


def test_the_detail_is_pruned_after_a_week_and_the_verdict_stays(tmp_path):
    validations = Validations(connect(str(tmp_path / "db")))
    week_ago = time.time() - URL_DETAIL_DAYS * 86400 - 60
    validations.record(verdict("old", week_ago), URLS)
    validations.record(verdict("new", time.time()), URLS)

    validations.prune_urls()

    assert validations.urls("old") == [] and validations.urls("new") == URLS
    assert validations.latest("old")["verdict"] == "pass"
    assert [t["task_id"] for t in validations.recent()] == ["new", "old"]
