from datetime import datetime, timezone

from neurons.validators.scoring_dataset import (
    HuggingFaceQuestionPool,
    build_scoring_assignments,
    filter_scoring_assignments,
)
from neurons.validators.seed_commitment import CommittedValidator


def build_validators():
    return [
        CommittedValidator(uid=1, hotkey="validator-a", seed=111),
        CommittedValidator(uid=2, hotkey="validator-b", seed=222),
    ]


def build_question_pool():
    pool = HuggingFaceQuestionPool(
        dataset_name="dummy",
        dataset_config=None,
        split="train",
        question_column="query",
    )
    pool._questions = [
        "what is bittensor",
        "what is tao",
        "what is subnet 22",
        "how does desearch work",
    ]
    return pool


def test_build_scoring_assignments_is_deterministic():
    time_range_start = datetime(2026, 4, 6, 10, 0, tzinfo=timezone.utc)
    pool = build_question_pool()

    assignments_a = build_scoring_assignments(
        time_range_start=time_range_start,
        miner_uids=[3, 7, 11],
        validators=build_validators(),
        question_pool=pool,
        combined_seed=123456,
    )
    assignments_b = build_scoring_assignments(
        time_range_start=time_range_start,
        miner_uids=[3, 7, 11],
        validators=build_validators(),
        question_pool=pool,
        combined_seed=123456,
    )

    assert assignments_a == assignments_b
    assert len(assignments_a) == 9
    assert sorted(item.uid for item in assignments_a).count(3) == 3
    assert sorted(item.uid for item in assignments_a).count(7) == 3
    assert sorted(item.uid for item in assignments_a).count(11) == 3
    assert sorted(item.search_type for item in assignments_a) == [
        "ai_search",
        "ai_search",
        "ai_search",
        "web_search",
        "web_search",
        "web_search",
        "x_search",
        "x_search",
        "x_search",
    ]

    owners_by_miner = {
        uid: {
            (item.validator_uid, item.validator_hotkey)
            for item in assignments_a
            if item.uid == uid
        }
        for uid in [3, 7, 11]
    }
    assert all(len(owners) == 1 for owners in owners_by_miner.values())

    local_assignments = filter_scoring_assignments(assignments_a, validator_uid=1)
    assert all(item.validator_uid == 1 for item in local_assignments)
