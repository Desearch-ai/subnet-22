"""The real validator, pointed at the local task API and a clock instead of the chain."""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace

import aiohttp
from bittensor.wallets import Keypair

from desearch.kinds import EMBED
from neurons.validators import validator as neuron
from neurons.validators.weights import EMISSION_CONTROL_HOTKEY

from sandbox.chain import FakeChain
from sandbox.local import BLOCK_SECONDS, TEMPO, VALIDATOR_URI, Ports


class SandboxValidator(neuron.Validator):
    def __init__(self, chain: FakeChain, embed: bool = False):
        super().__init__()
        self.chain = chain
        self.embed = embed

    async def initialize(self) -> None:
        self.wallet = SimpleNamespace(hotkey=Keypair.create_from_uri(VALIDATOR_URI))
        self.subtensor = self.chain
        self.metagraph = await self.chain.subnets.metagraph(self.config.netuid)
        self.uid = self.metagraph.hotkeys.index(self.wallet.hotkey.ss58_address)
        self.http = aiohttp.ClientSession(timeout=neuron.DOWNLOAD_TIMEOUT)

    def crawl_disabled_reason(self) -> str | None:
        # Only embed tasks are handed out in embed mode, so weights need no ScrapingDog key.
        return None if self.embed else super().crawl_disabled_reason()

    async def check_crawl_tasks(self) -> None:
        if self.embed:
            await self.stopping.wait()
            return
        await super().check_crawl_tasks()

    async def check_embed_tasks(self) -> None:
        if not self.embed:
            await self.stopping.wait()
            return
        await super().check_embed_tasks()

    async def weights(self):
        # The pretend chain registers every miner with paid work, so its share reaches the weights.
        paid = sorted({*self.ledger.crawl_paid(), *self.ledger.shares().get(EMBED, {})})
        self.chain.hotkeys += [
            miner for miner in paid if miner not in self.chain.hotkeys
        ]
        self.metagraph = await self.chain.subnets.metagraph(self.config.netuid)
        return await super().weights()


async def serve(validator: SandboxValidator) -> None:
    try:
        await validator.run()
    finally:
        await validator.stop()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m sandbox.validator")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--genesis", type=float, required=True)
    parser.add_argument("--embed", action="store_true")
    args = parser.parse_args(argv)
    ports, run_dir = Ports(args.port), args.run_dir
    # A sandbox epoch is a minute, so weights and the window report come every minute.
    neuron.POLL_S = 5
    neuron.AFTER_WEIGHTS_S = 30
    sys.argv = [
        "validator",
        "--netuid",
        "22",
        "--wandb.off",
        "--neuron.task_api_url",
        ports.api_url,
        "--neuron.storage_url",
        ports.storage_url,
        "--logging.logging_dir",
        str(run_dir / "logs" / "validator"),
        "--logging.info",
    ]
    hotkeys = [
        EMISSION_CONTROL_HOTKEY,
        Keypair.create_from_uri(VALIDATOR_URI).ss58_address,
    ]
    chain = FakeChain(hotkeys, args.genesis, BLOCK_SECONDS, TEMPO)
    asyncio.run(serve(SandboxValidator(chain, args.embed)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
