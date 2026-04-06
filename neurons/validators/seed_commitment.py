import asyncio
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
SEED_ENVELOPE_PATTERN = re.compile(r"<S:([^>]*)>")


@dataclass(frozen=True)
class SeedCommitment:
    time_range_start: datetime
    seed: int


@dataclass(frozen=True)
class CommittedValidator:
    uid: int
    hotkey: str
    seed: int


@dataclass(frozen=True)
class WindowSeedState:
    validator_count: int
    combined_seed: int


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

    committed_validators = await load_committed_validators(
        subtensor=subtensor,
        netuid=netuid,
        validators=validators,
        time_range_start=time_range_start,
    )

    return WindowSeedState(
        validator_count=len(committed_validators),
        combined_seed=combine_validator_seeds(
            committed_validators,
            time_range_start=time_range_start,
        ),
    )
