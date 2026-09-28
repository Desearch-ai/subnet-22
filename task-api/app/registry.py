from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass

from .auth import Keypair

RETRY_S = 60
DEV_RECEIPT_KEY = "//TaskApi"

log = logging.getLogger("task_api")


@dataclass
class Entry:
    hotkey: str
    uid: int | None
    is_validator: bool


class LocalRegistry:
    def __init__(self, validators: set[str] | None = None):
        self.validators = validators or set()

    async def start(self) -> None:
        pass

    async def stop(self) -> None:
        pass

    @classmethod
    def from_uris(cls, uris: str) -> LocalRegistry:
        return cls(
            {
                Keypair.create_from_uri(uri.strip()).ss58_address
                for uri in uris.split(",")
                if uri.strip()
            }
        )

    async def lookup(self, hotkey: str) -> Entry | None:
        return Entry(hotkey, None, hotkey in self.validators)


class ChainRegistry:
    """The subnet's hotkeys, refreshed on a timer so no request waits on the chain."""

    def __init__(
        self, netuid: int, network: str, ttl: int = 600, stake_threshold: float = 1000.0
    ):
        self.netuid = netuid
        self.network = network
        self.ttl = ttl
        self.stake_threshold = stake_threshold
        self._entries: dict[str, Entry] = {}
        self._loaded = asyncio.Event()
        self._refreshing: asyncio.Task | None = None

    async def start(self) -> None:
        if self._refreshing is None:
            self._refreshing = asyncio.create_task(self._refresh_forever())

    async def stop(self) -> None:
        if self._refreshing is not None:
            self._refreshing.cancel()
            self._refreshing = None

    async def lookup(self, hotkey: str) -> Entry | None:
        if not self._loaded.is_set():
            await self.start()
            await self._loaded.wait()
        return self._entries.get(hotkey)

    async def _refresh_forever(self) -> None:
        while True:
            try:
                self._entries = await asyncio.to_thread(self._load)
                self._loaded.set()
                wait = self.ttl
            except Exception:
                log.exception(
                    "metagraph refresh failed; keeping %d entries", len(self._entries)
                )
                wait = RETRY_S
            await asyncio.sleep(wait)

    def _load(self) -> dict[str, Entry]:
        import bittensor as bt

        chain = bt.Subtensor(self.network)
        try:
            metagraph = chain.subnets.metagraph(self.netuid)
        finally:
            chain.close()
        return {
            neuron.hotkey: Entry(
                neuron.hotkey,
                int(neuron.uid),
                bool(neuron.validator_permit)
                and float(neuron.total_stake.alpha) >= self.stake_threshold,
            )
            for neuron in metagraph.neurons
        }


def registry_from_env():
    mode = os.environ.get("TASK_API_REGISTRY", "")
    if mode == "chain":
        return ChainRegistry(
            netuid=int(os.environ.get("TASK_API_NETUID", "22")),
            network=os.environ.get("TASK_API_NETWORK", "finney"),
        )
    if mode == "local":
        return LocalRegistry.from_uris(os.environ.get("TASK_API_VALIDATOR_URIS", ""))
    raise RuntimeError("TASK_API_REGISTRY must be set to chain or local")


def admins_from_env() -> frozenset[str]:
    hotkeys = {
        hotkey.strip()
        for hotkey in os.environ.get("TASK_API_ADMIN_HOTKEYS", "").split(",")
        if hotkey.strip()
    }
    hotkeys |= LocalRegistry.from_uris(
        os.environ.get("TASK_API_ADMIN_URIS", "")
    ).validators
    return frozenset(hotkeys)


def receipt_key_from_env() -> Keypair:
    """On chain this must be a real secret, not the dev key."""
    uri = os.environ.get("TASK_API_KEY_URI", "")
    if not uri and os.environ.get("TASK_API_REGISTRY") == "chain":
        raise RuntimeError("TASK_API_KEY_URI must be set when TASK_API_REGISTRY=chain")
    return Keypair.create_from_uri(uri or DEV_RECEIPT_KEY)
