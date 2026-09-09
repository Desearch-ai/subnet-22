from __future__ import annotations

import hashlib
import os
import secrets
import time

BLOCK_SECONDS = 12
REVEAL_AFTER_BLOCKS = 2


class LocalSeeds:
    def __init__(self, delay: float = 2.0):
        self.delay = delay
        self._pending: dict[int, tuple[float, str]] = {}

    def target_block(self, opened_at: float) -> int:
        block = int(opened_at // BLOCK_SECONDS) + REVEAL_AFTER_BLOCKS
        self._pending.setdefault(block, (opened_at + self.delay, secrets.token_hex(32)))
        return block

    async def seed_for(self, block: int) -> str | None:
        entry = self._pending.get(block)
        if entry is None:
            return None
        available_at, value = entry
        return value if time.time() >= available_at else None


class ChainSeeds:
    def __init__(self, network: str = "finney"):
        self.network = network
        self._subtensor = None

    def _chain(self):
        if self._subtensor is None:
            import bittensor as bt

            self._subtensor = bt.subtensor(network=self.network)
        return self._subtensor

    def target_block(self, opened_at: float) -> int:
        return self._chain().get_current_block() + REVEAL_AFTER_BLOCKS

    async def seed_for(self, block: int) -> str | None:
        chain = self._chain()
        if chain.get_current_block() < block:
            return None
        return hashlib.sha256(str(chain.get_block_hash(block)).encode()).hexdigest()


def seeds_from_env():
    if os.environ.get("TASK_API_SEEDS", "local") == "chain":
        return ChainSeeds(os.environ.get("TASK_API_NETWORK", "finney"))
    return LocalSeeds(delay=float(os.environ.get("TASK_API_SEED_DELAY", "2")))
