import asyncio
import base64
import logging
import random
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Iterable

from neurons.validators.scoring_dataset import derive_deterministic_int

logger = logging.getLogger(__name__)

MAX_SEED_COMMITMENT_BYTES = 128
DEFAULT_SEED_COMMITMENT_REVEAL_DELAY_SECONDS = 30
DEFAULT_BUCKET_COMMITMENT_REVEAL_DELAY_SECONDS = 30
SEED_ENVELOPE_PATTERN = re.compile(r"<S:([^>]*)>")
BUCKET_ENVELOPE_PATTERN = re.compile(r"<B:([^>]*)>")


@dataclass(frozen=True)
class SeedCommitment:
    time_range_start: datetime
    seed: int


@dataclass(frozen=True)
class BucketCommitment:
    time_range_start: datetime
    bucket_locator: str


@dataclass(frozen=True)
class CommittedValidator:
    uid: int
    hotkey: str
    seed: int


@dataclass(frozen=True)
class CommittedValidatorBucket:
    uid: int
    hotkey: str
    bucket_locator: str


@dataclass(frozen=True)
class WindowSeedState:
    committed_validators: tuple[CommittedValidator, ...]
    combined_seed: int

    @property
    def validator_count(self) -> int:
        return len(self.committed_validators)


@dataclass(frozen=True)
class WindowBucketState:
    committed_buckets: tuple[CommittedValidatorBucket, ...]

    @property
    def validator_count(self) -> int:
        return len(self.committed_buckets)

    @property
    def bucket_locators(self) -> dict[tuple[int, str], str]:
        return {
            (bucket.uid, bucket.hotkey): bucket.bucket_locator
            for bucket in self.committed_buckets
        }


def extract_seed_payload(raw_commitment: str) -> tuple[str, str]:
    match = SEED_ENVELOPE_PATTERN.search(raw_commitment)
    if not match:
        return raw_commitment, ""

    seed_payload = match.group(1)
    other_data = SEED_ENVELOPE_PATTERN.sub("", raw_commitment).strip()
    return other_data, seed_payload


def merge_seed_commitment(raw_commitment: str, payload: str) -> str:
    if not raw_commitment:
        return f"<S:{payload}>"

    other_data, _ = extract_seed_payload(raw_commitment)
    envelope = f"<S:{payload}>"
    return f"{other_data}{envelope}" if other_data else envelope


def format_seed_payload(time_range_start: datetime, seed: int) -> str:
    timestamp = int(time_range_start.astimezone(timezone.utc).timestamp())
    return f"{timestamp}:{seed:x}"


def parse_seed_payload(payload: str) -> SeedCommitment | None:
    if not payload:
        return None

    try:
        timestamp_raw, seed_raw = payload.split(":", maxsplit=1)
        time_range_start = datetime.fromtimestamp(int(timestamp_raw), tz=timezone.utc)
        seed = int(seed_raw, 16)
        return SeedCommitment(time_range_start=time_range_start, seed=seed)
    except (TypeError, ValueError):
        logger.debug("Failed to parse seed commitment payload %r", payload)
        return None


def parse_seed_commitment(raw_commitment: str) -> SeedCommitment | None:
    _, payload = extract_seed_payload(raw_commitment)
    return parse_seed_payload(payload)


def extract_bucket_payload(raw_commitment: str) -> tuple[str, str]:
    match = BUCKET_ENVELOPE_PATTERN.search(raw_commitment)
    if not match:
        return raw_commitment, ""

    bucket_payload = match.group(1)
    other_data = BUCKET_ENVELOPE_PATTERN.sub("", raw_commitment).strip()
    return other_data, bucket_payload


def merge_bucket_commitment(raw_commitment: str, payload: str) -> str:
    if not raw_commitment:
        return f"<B:{payload}>"

    other_data, _ = extract_bucket_payload(raw_commitment)
    envelope = f"<B:{payload}>"
    return f"{other_data}{envelope}" if other_data else envelope


def _encode_bucket_locator(locator: str) -> str:
    return base64.urlsafe_b64encode(locator.encode("utf-8")).decode("ascii").rstrip("=")


def _decode_bucket_locator(encoded_locator: str) -> str:
    padding = "=" * (-len(encoded_locator) % 4)
    return base64.urlsafe_b64decode(
        f"{encoded_locator}{padding}".encode("ascii")
    ).decode("utf-8")


def format_bucket_payload(time_range_start: datetime, bucket_locator: str) -> str:
    timestamp = int(time_range_start.astimezone(timezone.utc).timestamp())
    return f"{timestamp}:{_encode_bucket_locator(bucket_locator)}"


def parse_bucket_payload(payload: str) -> BucketCommitment | None:
    if not payload:
        return None

    try:
        timestamp_raw, locator_raw = payload.split(":", maxsplit=1)
        time_range_start = datetime.fromtimestamp(int(timestamp_raw), tz=timezone.utc)
        bucket_locator = _decode_bucket_locator(locator_raw)
        return BucketCommitment(
            time_range_start=time_range_start,
            bucket_locator=bucket_locator,
        )
    except (TypeError, ValueError):
        logger.debug("Failed to parse bucket commitment payload %r", payload)
        return None


def parse_bucket_commitment(raw_commitment: str) -> BucketCommitment | None:
    _, payload = extract_bucket_payload(raw_commitment)
    return parse_bucket_payload(payload)


def combine_validator_seeds(
    validators: Iterable[CommittedValidator],
    *,
    time_range_start: datetime,
) -> int:
    ordered_fragments = [
        f"{validator.uid}:{validator.hotkey}:{validator.seed}"
        for validator in sorted(validators, key=lambda item: (item.uid, item.hotkey))
    ]
    return derive_deterministic_int(time_range_start.isoformat(), *ordered_fragments)


async def publish_seed_commitment(
    *,
    subtensor,
    wallet,
    netuid: int,
    uid: int,
    time_range_start: datetime,
    reveal_delay_seconds: int = DEFAULT_SEED_COMMITMENT_REVEAL_DELAY_SECONDS,
) -> SeedCommitment | None:
    current_raw_commitment = await subtensor.get_commitment(netuid, uid)

    current_commitment = parse_seed_commitment(current_raw_commitment or "")

    if current_commitment and current_commitment.time_range_start == time_range_start:
        logger.info(
            "Reusing existing seed commitment for window %s",
            time_range_start.isoformat(),
        )
        return current_commitment

    reveal_deadline = time_range_start + timedelta(seconds=reveal_delay_seconds)

    if datetime.now(timezone.utc) >= reveal_deadline:
        logger.warning(
            "Skipping seed commitment for %s because the reveal deadline passed.",
            time_range_start.isoformat(),
        )
        return None

    local_seed = random.SystemRandom().getrandbits(63)
    payload = format_seed_payload(time_range_start, local_seed)
    merged_commitment = merge_seed_commitment(current_raw_commitment or "", payload)

    commitment_bytes = len(merged_commitment.encode("utf-8"))

    if commitment_bytes > MAX_SEED_COMMITMENT_BYTES:
        raise ValueError(
            f"Seed commitment exceeds {MAX_SEED_COMMITMENT_BYTES} bytes: "
            f"{commitment_bytes}"
        )

    await subtensor.set_commitment(
        wallet=wallet,
        netuid=netuid,
        data=merged_commitment,
    )

    logger.info(
        "Published seed commitment for window %s",
        time_range_start.isoformat(),
    )

    return SeedCommitment(time_range_start=time_range_start, seed=local_seed)


async def wait_for_commitment_reveal(
    *,
    time_range_start: datetime,
    reveal_delay_seconds: int = DEFAULT_SEED_COMMITMENT_REVEAL_DELAY_SECONDS,
) -> None:
    reveal_deadline = time_range_start + timedelta(seconds=reveal_delay_seconds)
    remaining = (reveal_deadline - datetime.now(timezone.utc)).total_seconds()
    if remaining > 0:
        logger.info(
            "Waiting %.2fs for validator seed commitments for %s",
            remaining,
            time_range_start.isoformat(),
        )
        await asyncio.sleep(remaining)


async def publish_bucket_commitment(
    *,
    subtensor,
    wallet,
    netuid: int,
    uid: int,
    time_range_start: datetime,
    bucket_locator: str,
) -> BucketCommitment | None:
    current_raw_commitment = await subtensor.get_commitment(netuid, uid)
    current_commitment = parse_bucket_commitment(current_raw_commitment or "")

    if (
        current_commitment
        and current_commitment.time_range_start == time_range_start
        and current_commitment.bucket_locator == bucket_locator
    ):
        logger.info(
            "Reusing existing bucket commitment for window %s",
            time_range_start.isoformat(),
        )
        return current_commitment

    payload = format_bucket_payload(time_range_start, bucket_locator)
    merged_commitment = merge_bucket_commitment(current_raw_commitment or "", payload)

    commitment_bytes = len(merged_commitment.encode("utf-8"))
    if commitment_bytes > MAX_SEED_COMMITMENT_BYTES:
        raise ValueError(
            f"Bucket commitment exceeds {MAX_SEED_COMMITMENT_BYTES} bytes: "
            f"{commitment_bytes}"
        )

    await subtensor.set_commitment(
        wallet=wallet,
        netuid=netuid,
        data=merged_commitment,
    )

    logger.info(
        "Published bucket commitment for window %s",
        time_range_start.isoformat(),
    )

    return BucketCommitment(
        time_range_start=time_range_start,
        bucket_locator=bucket_locator,
    )


async def wait_for_bucket_commitment_reveal(
    *,
    time_range_start: datetime,
    publish_offset: timedelta,
    reveal_delay_seconds: int = DEFAULT_BUCKET_COMMITMENT_REVEAL_DELAY_SECONDS,
) -> None:
    reveal_deadline = (
        time_range_start + publish_offset + timedelta(seconds=reveal_delay_seconds)
    )
    remaining = (reveal_deadline - datetime.now(timezone.utc)).total_seconds()
    if remaining > 0:
        logger.info(
            "Waiting %.2fs for validator bucket commitments for %s",
            remaining,
            time_range_start.isoformat(),
        )
        await asyncio.sleep(remaining)


async def load_committed_validators(
    *,
    subtensor,
    netuid: int,
    validators,
    time_range_start: datetime,
) -> list[CommittedValidator]:
    raw_commitments = await subtensor.get_all_commitments(netuid)

    committed_validators: list[CommittedValidator] = []
    for validator in sorted(validators, key=lambda item: (item.uid, item.hotkey)):
        commitment = parse_seed_commitment(raw_commitments.get(validator.hotkey, ""))
        if commitment is None or commitment.time_range_start != time_range_start:
            continue

        committed_validators.append(
            CommittedValidator(
                uid=int(validator.uid),
                hotkey=validator.hotkey,
                seed=commitment.seed,
            )
        )

    logger.info(
        "Loaded %s committed validators for window %s",
        len(committed_validators),
        time_range_start.isoformat(),
    )
    return committed_validators


async def load_committed_validator_buckets(
    *,
    subtensor,
    netuid: int,
    validators,
    time_range_start: datetime,
) -> list[CommittedValidatorBucket]:
    raw_commitments = await subtensor.get_all_commitments(netuid)

    committed_buckets: list[CommittedValidatorBucket] = []
    for validator in sorted(validators, key=lambda item: (item.uid, item.hotkey)):
        commitment = parse_bucket_commitment(raw_commitments.get(validator.hotkey, ""))
        if commitment is None or commitment.time_range_start != time_range_start:
            continue

        committed_buckets.append(
            CommittedValidatorBucket(
                uid=int(validator.uid),
                hotkey=validator.hotkey,
                bucket_locator=commitment.bucket_locator,
            )
        )

    logger.info(
        "Loaded %s committed validator buckets for window %s",
        len(committed_buckets),
        time_range_start.isoformat(),
    )
    return committed_buckets


async def build_window_seed_state(
    *,
    subtensor,
    wallet,
    netuid: int,
    uid: int,
    validators,
    time_range_start: datetime,
    reveal_delay_seconds: int = DEFAULT_SEED_COMMITMENT_REVEAL_DELAY_SECONDS,
) -> WindowSeedState:
    await publish_seed_commitment(
        subtensor=subtensor,
        wallet=wallet,
        netuid=netuid,
        uid=uid,
        time_range_start=time_range_start,
        reveal_delay_seconds=reveal_delay_seconds,
    )

    await wait_for_commitment_reveal(
        time_range_start=time_range_start,
        reveal_delay_seconds=reveal_delay_seconds,
    )

    return await load_window_seed_state(
        subtensor=subtensor,
        netuid=netuid,
        validators=validators,
        time_range_start=time_range_start,
    )


async def load_window_seed_state(
    *,
    subtensor,
    netuid: int,
    validators,
    time_range_start: datetime,
) -> WindowSeedState:
    committed_validators = await load_committed_validators(
        subtensor=subtensor,
        netuid=netuid,
        validators=validators,
        time_range_start=time_range_start,
    )

    return WindowSeedState(
        committed_validators=tuple(committed_validators),
        combined_seed=combine_validator_seeds(
            committed_validators,
            time_range_start=time_range_start,
        ),
    )


async def build_window_bucket_state(
    *,
    subtensor,
    wallet,
    netuid: int,
    uid: int,
    validators,
    time_range_start: datetime,
    bucket_locator: str,
    publish_offset: timedelta,
    reveal_delay_seconds: int = DEFAULT_BUCKET_COMMITMENT_REVEAL_DELAY_SECONDS,
) -> WindowBucketState:
    await publish_bucket_commitment(
        subtensor=subtensor,
        wallet=wallet,
        netuid=netuid,
        uid=uid,
        time_range_start=time_range_start,
        bucket_locator=bucket_locator,
    )

    await wait_for_bucket_commitment_reveal(
        time_range_start=time_range_start,
        publish_offset=publish_offset,
        reveal_delay_seconds=reveal_delay_seconds,
    )

    return await load_window_bucket_state(
        subtensor=subtensor,
        netuid=netuid,
        validators=validators,
        time_range_start=time_range_start,
    )


async def load_window_bucket_state(
    *,
    subtensor,
    netuid: int,
    validators,
    time_range_start: datetime,
) -> WindowBucketState:
    committed_buckets = await load_committed_validator_buckets(
        subtensor=subtensor,
        netuid=netuid,
        validators=validators,
        time_range_start=time_range_start,
    )

    return WindowBucketState(committed_buckets=tuple(committed_buckets))
