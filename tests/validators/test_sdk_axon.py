import ipaddress

import bittensor.metagraph as sdk
import pytest

import neurons.validators as validators
from neurons.validators.journal import metagraph_rows

# Testnet netuid 41's legacy record: `ip_type` 4 with an ip past 2**32.
MALFORMED = {
    "block": 2067192,
    "version": 672,
    "ip": 196292440923238,
    "port": 0,
    "ip_type": 4,
    "protocol": 4,
    "placeholder1": 0,
    "placeholder2": 0,
}
SERVED = {"ip": int(ipaddress.IPv4Address("1.2.3.4")), "port": 8091, "ip_type": 4}


class Node:
    """A chain whose runtime answers `get_metagraph` with these axons, decoded as the SDK decodes it, and only what building one needs."""

    def __init__(self, axons: list):
        self.axons = axons

    async def at(self):
        return self

    async def runtime(self, method, params):
        return graph(self.axons)


def graph(axons: list) -> dict:
    return {
        "hotkeys": [f"5Hotkey{uid}" for uid in range(len(axons))],
        "coldkeys": [f"5Coldkey{uid}" for uid in range(len(axons))],
        "axons": axons,
        "validator_permit": [True] * len(axons),
        "total_stake": [500 * 10**9] * len(axons),
        "alpha_stake": [5 * 10**9] * len(axons),
        "block": 2067200,
        "tempo": 360,
        "last_step": 2067000,
        "blocks_since_last_step": 200,
        "owner_hotkey": "5Owner",
        "owner_coldkey": "5OwnerColdkey",
        "network_registered_at": 1,
    }


def test_the_shim_wraps_the_sdk_function_that_parses_axons():
    assert validators._sdk_axon_endpoint is not None, (
        "bittensor.metagraph._axon_endpoint is gone: the SDK moved or renamed it, so a malformed axon fails the metagraph again;"
        " point the shim in neurons/validators/__init__.py at its new place"
    )
    assert sdk._axon_endpoint is validators._axon_endpoint


async def test_a_malformed_axon_reads_as_none_and_the_metagraph_still_builds(
    monkeypatch,
):
    # What `subnets.metagraph(netuid, block, commitments=False)` runs once it has its block.
    built = await sdk.fetch(Node([SERVED, MALFORMED, None]), 41, commitments=False)
    assert [n.axon for n in built.neurons] == ["1.2.3.4:8091", None, None]
    assert [n.hotkey for n in built.neurons] == ["5Hotkey0", "5Hotkey1", "5Hotkey2"]
    assert [row[:4] for row in metagraph_rows(built)] == [
        [uid, f"5Hotkey{uid}", f"5Coldkey{uid}", True] for uid in range(3)
    ]

    # The SDK's own parser still fails the whole metagraph on that record, so the shim is what reads it.
    monkeypatch.setattr(sdk, "_axon_endpoint", validators._sdk_axon_endpoint)
    with pytest.raises(ipaddress.AddressValueError):
        await sdk.fetch(Node([SERVED, MALFORMED]), 41, commitments=False)
