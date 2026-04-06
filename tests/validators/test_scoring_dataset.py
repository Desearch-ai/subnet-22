from datetime import datetime, timezone

from neurons.validators.scoring_dataset import (
    HuggingFaceQuestionPool,
    build_scoring_assignments,
)


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
        question_pool=pool,
        combined_seed=123456,
    )
    assignments_b = build_scoring_assignments(
        time_range_start=time_range_start,
        miner_uids=[3, 7, 11],
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
