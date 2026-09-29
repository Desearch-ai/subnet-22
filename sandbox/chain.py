"""The slice of the chain client the validator uses, on a clock instead of a chain."""

from __future__ import annotations

import time
from types import SimpleNamespace

from desearch.manifest import local_block_hash


def current_block(genesis: float, block_seconds: float) -> int:
    return int((time.time() - genesis) / block_seconds)


class FakeChain:
    def __init__(
        self, hotkeys: list[str], genesis: float, block_seconds: float, tempo: int
    ):
        self.hotkeys = hotkeys
        self.genesis = genesis
        self.block_seconds = block_seconds
        self.tempo = tempo
        self.subnets = SimpleNamespace(metagraph=self.metagraph)
        self.neurons = SimpleNamespace(uid=self.uid)
        self.epochs = SimpleNamespace(
            blocks_until_next_epoch=self.blocks_until_next_epoch
        )
        self.hyperparameters = SimpleNamespace(
            min_allowed_weights=self.min_allowed_weights,
            max_weight_limit=self.max_weight_limit,
        )

    async def close(self) -> None:
        pass

    async def block(self) -> int:
        return current_block(self.genesis, self.block_seconds)

    async def block_info(self, block: int | None = None) -> SimpleNamespace:
        if block is None:
            block = await self.block()
        return SimpleNamespace(number=block, hash=local_block_hash(block))

    async def metagraph(self, netuid: int, block: int | None = None) -> SimpleNamespace:
        return SimpleNamespace(
            hotkeys=list(self.hotkeys), num_uids=len(self.hotkeys), tempo=self.tempo
        )

    async def uid(self, hotkey_ss58: str, netuid: int) -> int | None:
        return self.hotkeys.index(hotkey_ss58) if hotkey_ss58 in self.hotkeys else None

    async def blocks_until_next_epoch(self, netuid: int) -> int:
        block = await self.block()
        return self.tempo - (block + netuid + 1) % (self.tempo + 1)

    async def min_allowed_weights(self, netuid: int) -> int:
        return 1

    async def max_weight_limit(self, netuid: int) -> float:
        return 1.0

    async def execute(self, intent, wallet) -> SimpleNamespace:
        return SimpleNamespace(success=True, message="recorded locally", error=None)
