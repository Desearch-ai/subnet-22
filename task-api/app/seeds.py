from __future__ import annotations

import asyncio
import os
import time

from desearch.manifest import REVEAL_AFTER_BLOCKS, local_block_hash, seed_from_hash

BLOCK_SECONDS = 12.0

__all__ = ["REVEAL_AFTER_BLOCKS", "LocalSeeds", "ChainSeeds", "seeds_from_env"]


class LocalSeeds:
    def __init__(self, block_seconds: float = BLOCK_SECONDS, genesis: float = 0.0):
        self.block_seconds = block_seconds
        self.genesis = genesis

    async def current_block(self) -> int:
        return int((time.time() - self.genesis) // self.block_seconds)

    async def target_block(self) -> int:
        return await self.current_block() + REVEAL_AFTER_BLOCKS

    def wait_s(self) -> float:
        return REVEAL_AFTER_BLOCKS * self.block_seconds

    async def seed_for(self, block: int) -> str | None:
        if await self.current_block() < block:
            return None
        return seed_from_hash(local_block_hash(block))


class ChainSeeds:
    def __init__(self, network: str = "finney"):
        self.network = network
        self._subtensor = None

    def _chain(self):
        if self._subtensor is None:
            import bittensor as bt

            self._subtensor = bt.subtensor(network=self.network)
        return self._subtensor

    async def current_block(self) -> int:
        return await asyncio.to_thread(lambda: self._chain().get_current_block())

    async def target_block(self) -> int:
        return await self.current_block() + REVEAL_AFTER_BLOCKS

    def wait_s(self) -> float:
        return REVEAL_AFTER_BLOCKS * BLOCK_SECONDS

    async def seed_for(self, block: int) -> str | None:
        if await self.current_block() < block:
            return None
        found = await asyncio.to_thread(lambda: self._chain().get_block_hash(block))
        return seed_from_hash(found)


def seeds_from_env():
    if os.environ.get("TASK_API_SEEDS", "local") == "chain":
        return ChainSeeds(os.environ.get("TASK_API_NETWORK", "finney"))
    return LocalSeeds(
        float(os.environ.get("TASK_API_BLOCK_SECONDS", BLOCK_SECONDS)),
        float(os.environ.get("TASK_API_GENESIS", "0")),
    )
