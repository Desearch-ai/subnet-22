from collections import Counter

import pytest
from app.validations import Infeasible, build_vote, credited_urls, decide

JOB = {"urls": [f"https://site.example/{n}" for n in range(20)]}


def samples(**outcomes: int) -> list[dict]:
    urls = iter(JOB["urls"])
    return [
        {"url": next(urls), "outcome": outcome}
        for outcome, count in outcomes.items()
        for _ in range(count)
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


@pytest.mark.parametrize(
    "result, why",
    [
        ({"returned": 0, "samples": samples(matched=1)}, "more samples than rows"),
        ({"returned": 21, "samples": samples(matched=1)}, "more rows returned"),
        ({"returned": 16, "samples": samples(matched=2)}, "85%"),
        ({"returned": 20, "samples": samples(matched=1) * 2}, "sampled twice"),
        (
            {
                "returned": 20,
                "samples": [
                    {"url": "https://elsewhere.example/", "outcome": "matched"}
                ],
            },
            "not in the task",
        ),
        (
            {"returned": 20, "error_rows": 1, "samples": samples(errors_confirmed=2)},
            "more error samples",
        ),
        (
            {"returned": 20, "error_rows": 5, "samples": samples(matched=16)},
            "more content samples",
        ),
    ],
)
def test_a_report_the_task_could_not_have_produced_is_refused(result, why):
    with pytest.raises(Infeasible, match=why):
        build_vote(JOB, "v", {"verdict": "pass", **result})


def test_a_validators_own_failure_is_void_not_a_fail():
    result = {"verdict": "fail", "reason": "unscorable", "returned": 0, "samples": []}
    vote = build_vote(JOB, "v", result)

    assert (vote["verdict"], vote["credited"]) == ("void", 0)
    assert vote["result"]["reason"] == "unscorable"


def test_a_pass_the_validator_mostly_could_not_read_is_void():
    mostly = {"verdict": "pass", "returned": 20}
    vote = build_vote(
        JOB, "v", {**mostly, "samples": samples(matched=2, unverifiable=3)}
    )
    assert (vote["verdict"], vote["result"]["reason"]) == ("void", "unverifiable")

    still = build_vote(
        JOB, "v", {**mostly, "samples": samples(matched=3, unverifiable=2)}
    )
    assert (still["verdict"], still["credited"]) == ("pass", 20)


def vote(validator: str, credited: int, verdict: str = "pass") -> dict:
    result = {"verdict": verdict, "credited": credited}
    return {
        "validator": validator,
        "verdict": verdict,
        "credited": credited,
        "result": result,
    }


def test_passes_that_disagree_on_the_pay_need_a_third_opinion():
    honest, lowball = vote("a", 20), vote("b", 0)
    assert decide([honest, lowball]).outcome == "audit"

    overdue = decide([honest, lowball], overdue=True)
    assert (overdue.vote["validator"], overdue.agreed, overdue.disagreed) == (
        "b",
        [],
        [],
    )

    close = decide([honest, vote("b", 18)])
    assert (close.outcome, close.vote["credited"], close.agreed) == (
        "final",
        18,
        ["a", "b"],
    )


def test_the_pay_two_of_three_agree_on_wins_and_the_odd_one_out_is_marked():
    final = decide([vote("a", 20), vote("b", 0), vote("c", 19)])

    assert (final.vote["validator"], final.vote["credited"]) == ("c", 19)
    assert (final.agreed, final.disagreed) == (["a", "c"], ["b"])


def test_an_embed_pass_must_account_for_every_text():
    from app.validations import build_embed_vote

    job = {"texts": 6, "chars": 5400}
    matched = [{"text_id": f"t{i}", "outcome": "matched"} for i in range(3)]
    whole = {"verdict": "pass", "returned": 6, "matched": 3, "missing": 0}
    assert build_embed_vote(job, "v", {**whole, "samples": matched})["credited"] == 5400

    for broken, why in (
        ({"returned": 0, "samples": matched}, "more samples"),
        ({"returned": 6, "missing": 6, "samples": matched}, "every text embedded"),
        ({"returned": 6, "malformed": 6, "samples": matched}, "every text embedded"),
        ({"returned": 6, "samples": matched + matched[:1]}, "sampled twice"),
    ):
        with pytest.raises(Infeasible, match=why):
            build_embed_vote(job, "v", {**whole, **broken})


def test_three_votes_decide_even_when_two_passes_differ_on_the_pay():
    final = decide([vote("stamp", 100), vote("careful", 70), vote("strict", 0, "fail")])

    assert (final.outcome, final.vote["validator"], final.vote["credited"]) == (
        "final",
        "careful",
        70,
    )
    assert (final.agreed, final.disagreed) == (["careful"], ["strict", "stamp"])
