from __future__ import annotations

import asyncio
import logging
import os
import time
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
    def __init__(
        self, netuid: int, network: str, ttl: int = 600, stake_threshold: float = 1000.0
    ):
        self.netuid = netuid
        self.network = network
        self.ttl = ttl
        self.stake_threshold = stake_threshold
        self._entries: dict[str, Entry] = {}
        self._refreshed = 0.0
        self._refreshing: asyncio.Task | None = None

    async def lookup(self, hotkey: str) -> Entry | None:
        if time.time() - self._refreshed > self.ttl and self._refreshing is None:
            self._refreshing = asyncio.create_task(self._refresh())
        if not self._entries and self._refreshing is not None:
            await asyncio.shield(self._refreshing)
        return self._entries.get(hotkey)

    async def _refresh(self) -> None:
        try:
            self._entries = await asyncio.to_thread(self._load)
            self._refreshed = time.time()
        except Exception:
            log.exception(
                "metagraph refresh failed; keeping %d entries", len(self._entries)
            )
            self._refreshed = time.time() - self.ttl + RETRY_S
        finally:
            self._refreshing = None

    def _load(self) -> dict[str, Entry]:
        import bittensor as bt

        metagraph = bt.subtensor(network=self.network).metagraph(netuid=self.netuid)
        return {
            hotkey: Entry(
                hotkey, int(uid), bool(permit) and float(stake) >= self.stake_threshold
            )
            for uid, hotkey, stake, permit in zip(
                metagraph.uids,
                metagraph.hotkeys,
                metagraph.S,
                metagraph.validator_permit,
                strict=True,
            )
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
