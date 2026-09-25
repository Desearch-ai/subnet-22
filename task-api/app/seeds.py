from __future__ import annotations

import asyncio
import hashlib
import os
import secrets
import time

BLOCK_SECONDS = 12.0
# Far enough ahead to anchor the commitment before the seed block.
REVEAL_AFTER_BLOCKS = 10


class LocalSeeds:
    def __init__(self, block_seconds: float = BLOCK_SECONDS):
        self.block_seconds = block_seconds
        self._seeds: dict[int, str] = {}

    async def current_block(self) -> int:
        return int(time.time() // self.block_seconds)

    async def target_block(self) -> int:
        return await self.current_block() + REVEAL_AFTER_BLOCKS

    def wait_s(self) -> float:
        return REVEAL_AFTER_BLOCKS * self.block_seconds

    async def seed_for(self, block: int) -> str | None:
        if await self.current_block() < block:
            return None
        return self._seeds.setdefault(block, secrets.token_hex(32))


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
        return hashlib.sha256(str(found).encode()).hexdigest()


def seeds_from_env():
    if os.environ.get("TASK_API_SEEDS", "local") == "chain":
        return ChainSeeds(os.environ.get("TASK_API_NETWORK", "finney"))
    return LocalSeeds(float(os.environ.get("TASK_API_BLOCK_SECONDS", BLOCK_SECONDS)))
