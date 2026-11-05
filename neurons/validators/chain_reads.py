"""What a validator reads of its own node by block number: a block's hash and its extrinsics. A node keeps every finalized block
for good, unlike state, which a default (pruned) node keeps for the last 256 blocks only and answers "State discarded" for older,
so these reads work on any node, for any block the task API's log names."""

from __future__ import annotations

from typing import Any, Optional


async def rpc(client, method: str, params: list) -> Any:
    """A raw JSON-RPC call on the client's own connection; a stand-in chain answers through its own `rpc`."""
    own = getattr(client, "rpc", None)
    if own is not None:
        return await own(method, params)
    return await client._substrate.raw.rpc_request(method, params)


async def finalized_block(client) -> int:
    head = await rpc(client, "chain_getFinalizedHead", [])
    number = (await rpc(client, "chain_getHeader", [head]))["number"]
    return int(number, 16) if isinstance(number, str) else int(number)


async def block_hash(
    client, block: int, finalized: Optional[int] = None
) -> Optional[str]:
    """A finalized block's hash, None until it is finalized."""
    if block > (finalized if finalized is not None else await finalized_block(client)):
        return None
    return await rpc(client, "chain_getBlockHash", [block])


async def block_extrinsics(
    client, block: int, finalized: Optional[int] = None
) -> Optional[list[bytes]]:
    """The extrinsics of a finalized block as the chain encodes them, None until it is finalized."""
    found = await block_hash(client, block, finalized)
    if found is None:
        return None
    body = await rpc(client, "chain_getBlock", [found])
    extrinsics = ((body or {}).get("block") or {}).get("extrinsics") or []
    return [bytes.fromhex(str(e).removeprefix("0x")) for e in extrinsics]
