"""What the capacity test reads from and writes to the chain."""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

import bittensor as bt
from bittensor.wallets import Keypair


@dataclass(frozen=True)
class Commitment:
    block: int
    tag: bytes
    root: bytes


def _bytes(value: Any) -> bytes:
    if isinstance(value, (bytes, bytearray)):
        return bytes(value)
    if isinstance(value, str):
        return bytes.fromhex(value.removeprefix("0x"))
    return bytes(value)


def commitment_from(record: Any) -> Commitment | None:
    """A stored commitment in the test's shape, one Raw field and one Sha256 field; None for anything else."""
    if not isinstance(record, dict) or record.get("block") is None:
        return None
    info = record.get("info") or {}
    fields = info.get("fields") or []
    # The decoder nests the list one level on some runtimes.
    if len(fields) == 1 and isinstance(fields[0], (list, tuple)):
        fields = fields[0]
    tag = root = None
    for field in fields:
        if not isinstance(field, dict) or len(field) != 1:
            return None
        ((name, value),) = field.items()
        if name.startswith("Raw"):
            tag = _bytes(value)
        elif name == "Sha256":
            root = _bytes(value)
    if tag is None or root is None or len(root) != 32:
        return None
    return Commitment(int(record["block"]), tag, root)


class CapacityChain:
    def __init__(self, network: str, netuid: int, key: Keypair | None = None):
        self.network = network
        self.netuid = netuid
        self.key = key
        self.client: bt.Client | None = None

    async def __aenter__(self) -> CapacityChain:
        self.client = bt.Client(self.network)
        await self.client.__aenter__()
        return self

    async def __aexit__(self, *exc) -> None:
        if self.client is not None:
            await self.client.__aexit__(*exc)

    async def _rpc(self, method: str, params: list) -> Any:
        return await self.client._substrate.raw.rpc_request(method, params)

    async def heads(self) -> AsyncIterator[int]:
        """The number of each new best block, as soon as this node has it."""
        async for header in self.client.blocks():
            yield header.number

    async def block_hash(self, block: int) -> bytes:
        return _bytes(await self._rpc("chain_getBlockHash", [block]))

    async def finalized(self) -> int:
        head = await self._rpc("chain_getFinalizedHead", [])
        number = (await self._rpc("chain_getHeader", [head]))["number"]
        return int(number, 16) if isinstance(number, str) else int(number)

    async def commit(self, tag: bytes, root: bytes) -> None:
        """Sends the commitment and returns without waiting for its block; `commitment` tells whether it landed."""
        info = {"fields": [[{f"Raw{len(tag)}": tag}, {"Sha256": root}]]}
        call = bt.calls.Commitments.set_commitment(netuid=self.netuid, info=info)
        await self.client.submit_call(
            call, self.key, signer="hotkey", wait_for_inclusion=False, wait_for_finalization=False
        )

    async def commitment(self, hotkey: str, block: int | None = None) -> Commitment | None:
        """A hotkey's commitment as of a block, or as of the best block. A pruned node keeps state for about 256 blocks."""
        record = await self.client.query(("Commitments", "CommitmentOf"), [self.netuid, hotkey], block=block)
        return commitment_from(record)

    async def registered(self, block: int) -> dict[str, str]:
        """Each hotkey on the subnet at a block, with its coldkey."""
        graph = await self.client.subnets.metagraph(self.netuid, block=block, commitments=False)
        return {neuron.hotkey: neuron.coldkey for neuron in graph.neurons}
