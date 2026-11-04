import asyncio
from types import SimpleNamespace

import pytest

from neurons.validators import governance
from neurons.validators.governance import Approval, Neuron, Snapshot

H1, H2, H3 = "a" * 64, "b" * 64, "c" * 64
SIGNER = "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY"
OTHER = "5FHneW46xGXgs5mUiveU4sbTyGBzmstUspZC92UhjJM694ty"
NOW = 1_000_000
OLD = NOW - governance.TIMELOCK_BLOCKS
TAO = 10**9


def snapshot(
    commitments: dict, stakes=None, owner="owner-cold", owner_hotkey="owner-hot"
) -> Snapshot:
    stakes = stakes or {"v1": 40, "v2": 30, "v3": 30}
    neurons = [
        Neuron(hotkey, f"cold-{hotkey}", True, stake * TAO)
        for hotkey, stake in stakes.items()
    ]
    neurons += [
        Neuron("owner-hot", "owner-cold", True, 1000 * TAO),
        Neuron("owner-other", "owner-cold", True, 1000 * TAO),
        Neuron("miner", "cold-miner", False, 1000 * TAO),
    ]
    return Snapshot(NOW, tuple(neurons), owner, owner_hotkey, commitments)


@pytest.mark.parametrize(
    "text, expected",
    [
        (f"sn22tee:{H1}", Approval((H1,))),
        (f"sn22tee:{H1},{H2}", Approval((H1, H2))),
        (f"sn22tee:{H1}@{SIGNER}", Approval((H1,), SIGNER)),
        (f"sn22tee:{H1},{H2}@{SIGNER}", None),
        (f"sn22tee:{H1},{H2},{H3}", None),
        (f"sn22tee:{H1.upper()}", None),
        (f"sn22tee:{H1[:-1]}", None),
        (f"sn22tee:{H1}@not-an-address", None),
        (f"sn22:{H1}", None),
        ("some other subnet's model id", None),
        ("", None),
    ],
)
def test_only_the_three_commitment_forms_are_read(text, expected):
    assert governance.parse(text) == expected


def test_a_commitment_is_split_into_raw_fields_and_read_back_whole():
    text = governance.text_of(Approval((H1, H2)))
    fields = governance.raw_fields(text)
    assert all(len(field) <= 128 for field in fields) and len(fields) == 2
    encoded = [{f"Raw{len(f)}": "0x" + f.hex()} for f in fields]
    assert governance.decode_fields(encoded) == text
    assert governance.decode_fields([{"BigRaw": "0x" + text.encode().hex()}]) == text
    assert governance.decode_fields([{"Sha256": "0x" + "00" * 32}]) == ""


def test_the_owners_stake_and_validators_without_a_permit_never_count():
    everyone = {
        hotkey: (OLD, f"sn22tee:{H1}")
        for hotkey in ("owner-hot", "owner-other", "miner")
    }
    assert governance.support(snapshot(everyone), H1, 0) == 0
    assert not governance.approved(snapshot(everyone), H1)
    assert governance.total(snapshot({})) == 100 * TAO


def test_a_version_needs_more_than_half_the_independent_stake_for_the_whole_timelock():
    half = {"v2": (OLD, f"sn22tee:{H1}"), "v3": (OLD, f"sn22tee:{H1}")}
    assert not governance.approved(
        snapshot({"v3": half["v3"]}), H1
    ), "30 of 100 is not enough"
    exactly_half = snapshot(half, {"v1": 40, "v2": 20, "v3": 30, "v4": 10})
    assert governance.support(exactly_half, H1, governance.TIMELOCK_BLOCKS) == 50 * TAO
    assert not governance.approved(exactly_half, H1), "half is not more than half"
    assert governance.approved(snapshot(half), H1)


def test_a_commitment_younger_than_the_timelock_does_not_count_yet():
    young = {"v1": (OLD + 1, f"sn22tee:{H1}"), "v2": (OLD, f"sn22tee:{H1}")}
    assert (
        governance.support(snapshot(young), H1, governance.TIMELOCK_BLOCKS) == 30 * TAO
    )
    assert not governance.approved(snapshot(young), H1)
    assert governance.endorsed(snapshot(young), H1), "the running version needs no age"


def test_listing_a_successor_keeps_the_running_version_endorsed():
    moving = {hotkey: (NOW - 10, f"sn22tee:{H1},{H2}") for hotkey in ("v1", "v2", "v3")}
    assert governance.endorsed(snapshot(moving), H1)
    assert not governance.approved(
        snapshot(moving), H2
    ), "the successor waits out the timelock"
    aged = {hotkey: (OLD, f"sn22tee:{H1},{H2}") for hotkey in ("v1", "v2", "v3")}
    assert governance.approved(snapshot(aged), H2) and governance.approved(
        snapshot(aged), H1
    )


def test_appointing_counts_only_commitments_naming_that_signer():
    commitments = {
        "v1": (OLD, f"sn22tee:{H1}@{SIGNER}"),
        "v2": (OLD, f"sn22tee:{H1}@{OTHER}"),
        "v3": (OLD, f"sn22tee:{H1}"),
    }
    found = snapshot(commitments)
    assert not governance.appointed(found, H1, SIGNER)
    commitments["v3"] = (OLD, f"sn22tee:{H1}@{SIGNER}")
    assert governance.appointed(snapshot(commitments), H1, SIGNER)
    assert not governance.appointed(snapshot(commitments), H1, OTHER)
    assert governance.approved(
        snapshot(commitments), H1
    ), "an appointment approves its version too"


def test_with_no_independent_stake_nothing_is_approved():
    assert not governance.approved(snapshot({}, {}), H1)
    assert not governance.endorsed(snapshot({}, {}), H1)


def test_the_snapshot_is_read_from_the_metagraph_at_the_asked_block():
    def neuron(uid, hotkey, coldkey, permit, stake):
        return SimpleNamespace(
            uid=uid,
            hotkey=hotkey,
            coldkey=coldkey,
            validator_permit=permit,
            total_stake=SimpleNamespace(rao=stake),
        )

    text = f"sn22tee:{H1}"
    graph = SimpleNamespace(
        block=1234,
        owner_coldkey="owner-cold",
        owner_hotkey="owner-hot",
        neurons=[neuron(0, "v1", "c1", True, 5), neuron(1, "m", "c2", False, 7)],
        commitments={
            0: SimpleNamespace(
                block=1000, fields=[{f"Raw{len(text)}": "0x" + text.encode().hex()}]
            )
        },
    )
    asked = []

    class Subnets:
        async def metagraph(self, netuid, block=None, *, commitments=True):
            asked.append((netuid, block, commitments))
            return graph

    found = asyncio.run(
        governance.read_snapshot(SimpleNamespace(subnets=Subnets()), 22, 1234)
    )
    assert asked == [(22, 1234, True)]
    assert found.block == 1234 and found.commitments == {"v1": (1000, text)}
    assert found.neurons[0] == Neuron("v1", "c1", True, 5)
    assert (found.owner_coldkey, found.owner_hotkey) == ("owner-cold", "owner-hot")
