from collections import Counter

import pytest
from app.validations import build_vote, credited_urls

JOB = {"urls": [f"https://site.example/{n}" for n in range(20)]}


def samples(**outcomes: int) -> list[dict]:
    return [
        {"url": f"https://site.example/{outcome}/{n}", "outcome": outcome}
        for outcome, count in outcomes.items()
        for n in range(count)
    ]


@pytest.mark.parametrize(
    "ok_rows, error_rows, outcomes, credited",
    [
        (16, 4, {"matched": 3, "errors_unconfirmed": 2}, 16),
        (16, 4, {"matched": 3, "errors_confirmed": 2}, 20),
        (16, 4, {"matched": 3, "errors_confirmed": 1, "errors_unconfirmed": 1}, 18),
        (16, 4, {"matched": 3, "not_fetched": 2}, 16),
        (20, 0, {"matched": 4, "mismatched": 1}, 16),
        (0, 10, {"errors_confirmed": 5}, 10),
        (20, 0, {"unverifiable": 5}, 0),
    ],
)
def test_a_task_is_paid_at_its_samples_rate(ok_rows, error_rows, outcomes, credited):
    assert credited_urls(ok_rows, error_rows, Counter(outcomes)) == credited


@pytest.mark.parametrize("matched, mismatched", [(5, 0), (3, 2), (0, 5)])
def test_credit_never_exceeds_what_was_returned(matched, mismatched):
    outcomes = Counter(matched=matched, mismatched=mismatched, errors_confirmed=2)
    assert credited_urls(15, 5, outcomes) <= 20


def test_a_failed_task_is_paid_nothing():
    result = {"verdict": "fail", "returned": 20, "samples": samples(matched=5)}
    assert build_vote(JOB, "v", result)["credited"] == 0


def test_a_pass_without_evidence_is_void():
    result = {"verdict": "pass", "returned": 20, "samples": samples(unverifiable=5)}
    vote = build_vote(JOB, "v", result)
    assert (vote["verdict"], vote["credited"]) == ("void", 0)


def test_the_validators_own_credit_is_ignored():
    result = {
        "verdict": "pass",
        "returned": 20,
        "credited": 999,
        "samples": samples(matched=4, mismatched=1),
    }
    assert build_vote(JOB, "v", result)["credited"] == 16
