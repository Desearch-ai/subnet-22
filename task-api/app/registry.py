from __future__ import annotations

import os
import time
from dataclasses import dataclass


@dataclass
class Entry:
    hotkey: str
    uid: int | None
    is_validator: bool


class LocalRegistry:
    def __init__(self, validators: set[str] | None = None):
        self.validators = validators or set()

    async def lookup(self, hotkey: str) -> Entry | None:
        return Entry(hotkey, None, hotkey in self.validators)


class ChainRegistry:
    def __init__(self, netuid: int, network: str, ttl: int = 600, stake_threshold: float = 1000.0):
        self.netuid = netuid
        self.network = network
        self.ttl = ttl
        self.stake_threshold = stake_threshold
        self._entries: dict[str, Entry] = {}
        self._refreshed = 0.0

    async def lookup(self, hotkey: str) -> Entry | None:
        if time.time() - self._refreshed > self.ttl:
            self._refresh()
        return self._entries.get(hotkey)

    def _refresh(self) -> None:
        import bittensor as bt

        metagraph = bt.subtensor(network=self.network).metagraph(netuid=self.netuid)
        self._entries = {
            hotkey: Entry(hotkey, int(uid), float(stake) >= self.stake_threshold)
            for uid, hotkey, stake in zip(metagraph.uids, metagraph.hotkeys, metagraph.S)
        }
        self._refreshed = time.time()


def registry_from_env():
    if os.environ.get("TASK_API_REGISTRY", "local") == "chain":
        return ChainRegistry(
            netuid=int(os.environ.get("TASK_API_NETUID", "22")),
            network=os.environ.get("TASK_API_NETWORK", "finney"),
        )
    return LocalRegistry()
