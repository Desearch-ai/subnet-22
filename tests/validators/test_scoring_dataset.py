from collections import Counter
from datetime import datetime, timezone

from neurons.validators.service.scoring_dataset import (
    HuggingFaceQuestionPool,
    build_scoring_assignments,
    build_validator_ownership,
    filter_scoring_assignments,
)
from neurons.validators.service.seed_commitment import CommittedValidator


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


def test_build_validator_ownership_is_stable_for_same_scoring_window():
    time_range_start = datetime(2026, 4, 6, 10, 0, tzinfo=timezone.utc)
    miner_uids = [3, 7, 11, 13, 17, 19]
    validators = build_validators()

    ownership_a = build_validator_ownership(
        time_range_start=time_range_start,
        miner_uids=miner_uids,
        validators=validators,
        combined_seed=123456,
    )
    ownership_b = build_validator_ownership(
        time_range_start=time_range_start,
        miner_uids=list(reversed(miner_uids)),
        validators=list(reversed(validators)),
        combined_seed=123456,
    )

    assert ownership_a == ownership_b
    assert set(ownership_a) == set(miner_uids)

    owned_counts = Counter(owner.uid for owner in ownership_a.values())
    assert set(owned_counts) == {1, 2}
    assert max(owned_counts.values()) - min(owned_counts.values()) <= 1


def test_two_validators_split_miner_uids_without_overlap():
    time_range_start = datetime(2026, 4, 6, 10, 0, tzinfo=timezone.utc)
    miner_uids = [3, 7, 11, 13, 17, 19]
    assignments = build_scoring_assignments(
        time_range_start=time_range_start,
        miner_uids=miner_uids,
        validators=build_validators(),
        question_pool=build_question_pool(),
        combined_seed=123456,
    )

    validator_1_assignments = filter_scoring_assignments(assignments, validator_uid=1)
    validator_2_assignments = filter_scoring_assignments(assignments, validator_uid=2)

    validator_1_miner_uids = {item.uid for item in validator_1_assignments}
    validator_2_miner_uids = {item.uid for item in validator_2_assignments}

    assert validator_1_miner_uids
    assert validator_2_miner_uids
    assert validator_1_miner_uids.isdisjoint(validator_2_miner_uids)
    assert validator_1_miner_uids | validator_2_miner_uids == set(miner_uids)

    assert len(validator_1_assignments) == len(validator_1_miner_uids) * 3
    assert len(validator_2_assignments) == len(validator_2_miner_uids) * 3
