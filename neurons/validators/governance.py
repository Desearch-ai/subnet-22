"""Which task API version validators' stake has approved, from their commitments on chain (docs/tee.md).

Independent validators hold a permit, are not the subnet owner's coldkey or hotkey, and inherit no stake from a hotkey of the
owner's; what the owner's coldkey delegated to one does not count for it. A version needs more than the threshold of their stake
and at least `MIN_COLDKEYS` distinct coldkeys among them."""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass, field
from fractions import Fraction

from bittensor.wallets import Keypair

PREFIX = "sn22tee:"
# Strictly more than this share of the independent validators' stake approves.
THRESHOLD = Fraction(1, 2)
# Blocks a commitment must be on chain before it counts for a new version: 7 days.
TIMELOCK_BLOCKS = 50_400
# Distinct coldkeys among the approving validators, however much stake fewer of them hold.
MIN_COLDKEYS = 3
ROOT_NETUID = 0
U64_MAX = 2**64 - 1
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
    # hotkey -> the stake the owner's coldkey holds on it, as the metagraph weighs it
    owner_stake: dict[str, int] = field(default_factory=dict)
    # hotkeys a hotkey of the owner passes stake down to as a parent
    owner_children: frozenset[str] = frozenset()


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
    """Validators with a permit that the subnet owner neither holds nor passes stake down to."""
    return [
        n
        for n in snapshot.neurons
        if n.permit
        and n.coldkey != snapshot.owner_coldkey
        and n.hotkey != snapshot.owner_hotkey
        and n.hotkey not in snapshot.owner_children
    ]


def weight(snapshot: Snapshot, neuron: Neuron) -> int:
    """A validator's stake without what the owner's coldkey delegated to it."""
    return max(0, neuron.stake - snapshot.owner_stake.get(neuron.hotkey, 0))


def total(snapshot: Snapshot) -> int:
    return sum(weight(snapshot, n) for n in independent(snapshot))


def supporters(
    snapshot: Snapshot, version: str, age: int, signer: str | None = None
) -> list[Neuron]:
    """Independent validators whose commitment lists the version, made at least `age` blocks ago (and appoints `signer`)."""
    found = []
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
            found.append(neuron)
    return found


def support(
    snapshot: Snapshot, version: str, age: int, signer: str | None = None
) -> int:
    """Their stake."""
    return sum(weight(snapshot, n) for n in supporters(snapshot, version, age, signer))


def coldkeys(
    snapshot: Snapshot, version: str, age: int, signer: str | None = None
) -> int:
    """Their distinct coldkeys."""
    return len({n.coldkey for n in supporters(snapshot, version, age, signer)})


def passes(
    snapshot: Snapshot,
    version: str,
    age: int,
    signer: str | None,
    threshold: Fraction,
    min_coldkeys: int,
) -> bool:
    whole = total(snapshot)
    stake = support(snapshot, version, age, signer)
    return (
        whole > 0
        and stake > threshold * whole
        and coldkeys(snapshot, version, age, signer) >= min_coldkeys
    )


def least(min_coldkeys: int | None) -> int:
    """The coldkeys asked for, else the release's."""
    return MIN_COLDKEYS if min_coldkeys is None else min_coldkeys


def approved(
    snapshot: Snapshot,
    version: str,
    threshold: Fraction = THRESHOLD,
    timelock: int = TIMELOCK_BLOCKS,
    min_coldkeys: int | None = None,
) -> bool:
    """A new version may take over: approved for at least the timelock."""
    return passes(snapshot, version, timelock, None, threshold, least(min_coldkeys))


def appointed(
    snapshot: Snapshot,
    version: str,
    signer: str,
    threshold: Fraction = THRESHOLD,
    timelock: int = TIMELOCK_BLOCKS,
    min_coldkeys: int | None = None,
) -> bool:
    """One CVM of a version may start the custody chain anew."""
    return passes(snapshot, version, timelock, signer, threshold, least(min_coldkeys))


def endorsed(
    snapshot: Snapshot,
    version: str,
    threshold: Fraction = THRESHOLD,
    min_coldkeys: int | None = None,
) -> bool:
    """The running version is still listed by enough stake, however recently."""
    return passes(snapshot, version, 0, None, threshold, least(min_coldkeys))


def stake_by_hotkey(positions, netuid: int, tao_weight: int) -> dict[str, int]:
    """A coldkey's positions summed per hotkey as the metagraph's total stake counts them: alpha on the subnet, root TAO at
    `tao_weight` of `U64_MAX`."""
    found: dict[str, int] = {}
    for position in positions:
        on, rao = int(position.netuid), int(position.stake.rao)
        if on == netuid:
            counted = rao
        elif on == ROOT_NETUID:
            counted = rao * int(tao_weight) // U64_MAX
        else:
            continue
        found[str(position.hotkey)] = found.get(str(position.hotkey), 0) + counted
    return found


async def owner_children(
    view, netuid: int, hotkeys: list[str], owner_coldkey: str, owner_hotkey: str
) -> frozenset[str]:
    """The hotkeys among `hotkeys` that a hotkey of the owner passes stake down to as a parent."""
    parents = await asyncio.gather(
        *(view.delegation.parents(hotkey, netuid) for hotkey in hotkeys)
    )
    named = sorted({parent for found in parents for _, parent in found})
    coldkeys = dict(
        zip(
            named,
            await asyncio.gather(*(view.neurons.hotkey_owner(p) for p in named)),
        )
    )

    def owned(parent: str) -> bool:
        return parent == owner_hotkey or (
            bool(owner_coldkey) and coldkeys.get(parent) == owner_coldkey
        )

    return frozenset(
        hotkey
        for hotkey, found in zip(hotkeys, parents)
        if any(owned(parent) for _, parent in found)
    )


async def read_snapshot(subtensor, netuid: int, block: int | None = None) -> Snapshot:
    """The metagraph, commitments and the owner's stake at one block; old blocks are read from an archive node."""
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
    view = await subtensor.at(int(graph.block))
    owner_coldkey, owner_hotkey = str(graph.owner_coldkey), str(graph.owner_hotkey)
    owner_stake = {}
    if owner_coldkey:
        positions, tao_weight = await asyncio.gather(
            view.staking.stake_for_coldkey(owner_coldkey),
            view.query(("SubtensorModule", "TaoWeight")),
        )
        owner_stake = stake_by_hotkey(positions, netuid, int(tao_weight or 0))
    return Snapshot(
        block=int(graph.block),
        neurons=neurons,
        owner_coldkey=owner_coldkey,
        owner_hotkey=owner_hotkey,
        commitments=commitments,
        owner_stake=owner_stake,
        owner_children=await owner_children(
            view,
            netuid,
            [n.hotkey for n in neurons if n.permit],
            owner_coldkey,
            owner_hotkey,
        ),
    )
