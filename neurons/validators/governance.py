"""Which task API version validators' stake has approved, from their commitments on chain (docs/tee.md)."""

from __future__ import annotations

import re
from dataclasses import dataclass
from fractions import Fraction

from bittensor.wallets import Keypair

PREFIX = "sn22tee:"
# Strictly more than this share of the independent validators' stake approves.
THRESHOLD = Fraction(1, 2)
# Blocks a commitment must be on chain before it counts for a new version: 7 days.
TIMELOCK_BLOCKS = 50_400
HASH = re.compile(r"^[0-9a-f]{64}$")
RAW_FIELD = 128


@dataclass(frozen=True)
class Approval:
    versions: tuple[str, ...]
    signer: str | None = None


@dataclass(frozen=True)
class Neuron:
    hotkey: str
    coldkey: str
    permit: bool
    stake: int


@dataclass(frozen=True)
class Snapshot:
    """What approval is computed from, all at one block."""

    block: int
    neurons: tuple[Neuron, ...]
    owner_coldkey: str
    owner_hotkey: str
    # hotkey -> (block the commitment was made at, its text)
    commitments: dict[str, tuple[int, str]]


def is_ss58(address: str) -> bool:
    try:
        Keypair(ss58_address=address)
        return True
    except Exception:
        return False


def parse(text: str) -> Approval | None:
    """The approval a commitment states, None for anything else."""
    if not isinstance(text, str) or not text.startswith(PREFIX):
        return None
    body = text[len(PREFIX) :]
    signer = None
    if "@" in body:
        body, signer = body.split("@", 1)
        if not is_ss58(signer):
            return None
    versions = tuple(body.split(","))
    if not 1 <= len(versions) <= 2 or not all(HASH.match(h) for h in versions):
        return None
    if signer and len(versions) != 1:
        return None
    return Approval(versions, signer)


def text_of(approval: Approval) -> str:
    text = PREFIX + ",".join(approval.versions)
    return f"{text}@{approval.signer}" if approval.signer else text


def raw_fields(text: str) -> list[bytes]:
    """The commitment split into the Raw fields of at most 128 bytes the chain takes."""
    data = text.encode()
    return [data[i : i + RAW_FIELD] for i in range(0, len(data), RAW_FIELD)]


def decode_fields(fields: list) -> str:
    """The concatenated Raw and BigRaw bytes of a commitment, as text."""
    data = b""
    for entry in fields or []:
        for variant, value in (entry or {}).items():
            if (variant.startswith("Raw") or variant == "BigRaw") and isinstance(
                value, str
            ):
                data += bytes.fromhex(value.removeprefix("0x"))
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return ""


def independent(snapshot: Snapshot) -> list[Neuron]:
    """Validators with a permit that are not the subnet owner's."""
    return [
        n
        for n in snapshot.neurons
        if n.permit
        and n.coldkey != snapshot.owner_coldkey
        and n.hotkey != snapshot.owner_hotkey
    ]


def total(snapshot: Snapshot) -> int:
    return sum(n.stake for n in independent(snapshot))


def support(
    snapshot: Snapshot, version: str, age: int, signer: str | None = None
) -> int:
    """Stake whose commitment lists the version, made at least `age` blocks ago (and appoints `signer`)."""
    found = 0
    for neuron in independent(snapshot):
        committed = snapshot.commitments.get(neuron.hotkey)
        if committed is None:
            continue
        block, text = committed
        approval = parse(text)
        if approval is None or version not in approval.versions:
            continue
        if signer is not None and approval.signer != signer:
            continue
        if snapshot.block - block >= age:
            found += neuron.stake
    return found


def passes(snapshot: Snapshot, stake: int, threshold: Fraction) -> bool:
    whole = total(snapshot)
    return whole > 0 and stake > threshold * whole


def approved(
    snapshot: Snapshot,
    version: str,
    threshold: Fraction = THRESHOLD,
    timelock: int = TIMELOCK_BLOCKS,
) -> bool:
    """A new version may take over: approved for at least the timelock."""
    return passes(snapshot, support(snapshot, version, timelock), threshold)


def appointed(
    snapshot: Snapshot,
    version: str,
    signer: str,
    threshold: Fraction = THRESHOLD,
    timelock: int = TIMELOCK_BLOCKS,
) -> bool:
    """One CVM of a version may start the custody chain anew."""
    return passes(snapshot, support(snapshot, version, timelock, signer), threshold)


def endorsed(snapshot: Snapshot, version: str, threshold: Fraction = THRESHOLD) -> bool:
    """The running version is still listed by enough stake, however recently."""
    return passes(snapshot, support(snapshot, version, 0), threshold)


async def read_snapshot(subtensor, netuid: int, block: int | None = None) -> Snapshot:
    """The metagraph and commitments at one block; old blocks are read from an archive node."""
    graph = await subtensor.subnets.metagraph(netuid, block, commitments=True)
    if graph is None:
        raise ValueError(f"subnet {netuid} does not exist at block {block}")
    neurons = tuple(
        Neuron(
            hotkey=str(n.hotkey),
            coldkey=str(n.coldkey),
            permit=bool(n.validator_permit),
            stake=int(n.total_stake.rao),
        )
        for n in graph.neurons
    )
    commitments = {}
    for neuron in graph.neurons:
        found = graph.commitments.get(neuron.uid)
        if found is not None:
            commitments[str(neuron.hotkey)] = (
                int(found.block),
                decode_fields(found.fields),
            )
    return Snapshot(
        block=int(graph.block),
        neurons=neurons,
        owner_coldkey=str(graph.owner_coldkey),
        owner_hotkey=str(graph.owner_hotkey),
        commitments=commitments,
    )
