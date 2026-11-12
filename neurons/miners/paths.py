"""Several ways to the task API: directly, and through validators' relays.

Each request goes out on a path picked at random among those that worked lately, and moves on to the next when
one does not answer, so neither the network in front of the API nor any one relay can single a miner out by where
its requests come from, and blocking one path only costs the miner a retry. A relay signs a witness of every
request it forwards; the miner keeps those, and can prove with them that a request it sent never reached the log.
"""

from __future__ import annotations

import json
import logging
import random
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from bittensor.wallets import Keypair

from desearch.client import TaskApiClient, TaskApiError

REST_S = 30.0
MAX_REST_S = 600.0

log = logging.getLogger("miner")


def path_trouble(exc: TaskApiError) -> bool:
    """No answer, a rate limit or a server error: another path may get through where this one did not."""
    return exc.status == 0 or exc.status == 429 or exc.status >= 500


@dataclass
class Path:
    name: str
    client: TaskApiClient
    failures: int = 0
    resting_until: float = 0.0

    def failed(self, now: float) -> None:
        self.failures += 1
        self.resting_until = now + min(MAX_REST_S, REST_S * 2 ** (self.failures - 1))

    def worked(self) -> None:
        self.failures, self.resting_until = 0, 0.0


class MultiPathClient:
    """The `TaskApiClient` interface over several paths; every attempt is signed afresh."""

    def __init__(
        self,
        key: Keypair | str,
        direct: str | None,
        relays: Sequence[str] = (),
        timeout: float = 30.0,
        on_witness: Callable[[dict], None] | None = None,
        rng: random.Random | None = None,
    ):
        keypair = Keypair.create_from_uri(key) if isinstance(key, str) else key
        urls = ([("direct", direct)] if direct else []) + [
            (f"relay {url}", url) for url in relays
        ]
        if not urls:
            raise ValueError(
                "no path to the task API: give it directly or at least one relay"
            )
        self.paths = [
            Path(name, TaskApiClient(url, keypair, timeout, on_witness=on_witness))
            for name, url in urls
        ]
        self.hotkey: str = keypair.ss58_address
        self.rng = rng or random.SystemRandom()

    def order(self, now: float) -> list[Path]:
        """Paths that worked lately in a random order, then the resting ones, soonest back first."""
        ready = [p for p in self.paths if p.resting_until <= now]
        self.rng.shuffle(ready)
        resting = sorted(
            (p for p in self.paths if p.resting_until > now),
            key=lambda p: p.resting_until,
        )
        return ready + resting

    async def post(self, path: str, body: dict | None = None) -> dict:
        return await self._send(lambda client: client.post(path, body))

    async def get(self, path: str) -> dict:
        return await self._send(lambda client: client.get(path))

    async def _send(self, call) -> dict:
        last: TaskApiError | None = None
        for way in self.order(time.monotonic()):
            try:
                answer = await call(way.client)
            except TaskApiError as exc:
                if not path_trouble(exc):
                    way.worked()
                    raise
                way.failed(time.monotonic())
                last = exc
                log.warning(
                    "task API via %s failed (%s), trying another path", way.name, exc
                )
                continue
            way.worked()
            return answer
        assert last is not None
        raise last

    async def aclose(self) -> None:
        for way in self.paths:
            await way.client.aclose()

    async def __aenter__(self) -> MultiPathClient:
        return self

    async def __aexit__(self, *_) -> None:
        await self.aclose()


def keep_witness(path: str) -> Callable[[dict], None]:
    """Appends each witness a relay signed for us to a file, one JSON object per line."""

    def keep(record: dict) -> None:
        try:
            with open(path, "a") as handle:
                handle.write(json.dumps(record, sort_keys=True) + "\n")
        except (OSError, ValueError) as exc:
            log.warning("could not keep a relay's witness: %s", exc)

    return keep
