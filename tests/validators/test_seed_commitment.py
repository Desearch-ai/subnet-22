from datetime import datetime, timezone

from neurons.validators.service.seed_commitment import (
    CommittedValidator,
    combine_validator_seeds,
    format_bucket_payload,
    format_seed_payload,
    merge_bucket_commitment,
    merge_seed_commitment,
    parse_bucket_commitment,
    parse_seed_commitment,
)


def test_seed_commitment_round_trip_preserves_other_data():
    time_range_start = datetime(2026, 4, 6, 10, 0, tzinfo=timezone.utc)
    payload = format_seed_payload(time_range_start, 0xABC123)
    merged = merge_seed_commitment("https://example.com/data", payload)

    parsed = parse_seed_commitment(merged)

    assert merged.startswith("https://example.com/data")
    assert parsed is not None
    assert parsed.time_range_start == time_range_start
    assert parsed.seed == 0xABC123


def test_combine_validator_seeds_is_stable_for_same_validators():
    time_range_start = datetime(2026, 4, 6, 11, 0, tzinfo=timezone.utc)
    validators = [
        CommittedValidator(
            uid=5,
            hotkey="validator-b",
            seed=222,
        ),
        CommittedValidator(
            uid=1,
            hotkey="validator-a",
            seed=111,
        ),
    ]

    combined_a = combine_validator_seeds(
        validators,
        time_range_start=time_range_start,
    )
    combined_b = combine_validator_seeds(
        list(reversed(validators)),
        time_range_start=time_range_start,
    )

    assert combined_a == combined_b


def test_bucket_commitment_round_trip_preserves_other_data():
    time_range_start = datetime(2026, 4, 6, 10, 30, tzinfo=timezone.utc)
    payload = format_bucket_payload(time_range_start, "hf:namespace/repo-name")
    merged = merge_bucket_commitment("https://example.com/data", payload)

    parsed = parse_bucket_commitment(merged)

    assert merged.startswith("https://example.com/data")
    assert parsed is not None
    assert parsed.time_range_start == time_range_start
    assert parsed.bucket_locator == "hf:namespace/repo-name"
