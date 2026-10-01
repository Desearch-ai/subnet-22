"""The real validator, pointed at the local task API and a clock instead of the chain."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace

import aiohttp
from bittensor.wallets import Keypair

from neurons.validators import validator as neuron
from neurons.validators.weights import EMISSION_CONTROL_HOTKEY

from sandbox.chain import FakeChain
from sandbox.local import BLOCK_SECONDS, TEMPO, VALIDATOR_URI, Ports


class SandboxValidator(neuron.Validator):
    def __init__(self, chain: FakeChain):
        super().__init__()
        self.chain = chain

    async def initialize(self) -> None:
        self.wallet = SimpleNamespace(hotkey=Keypair.create_from_uri(VALIDATOR_URI))
        self.subtensor = self.chain
        self.metagraph = await self.chain.subnets.metagraph(self.config.netuid)
        self.uid = self.metagraph.hotkeys.index(self.wallet.hotkey.ss58_address)
        self.http = aiohttp.ClientSession(timeout=neuron.DOWNLOAD_TIMEOUT)


async def serve(validator: SandboxValidator) -> None:
    try:
        await validator.run()
    finally:
        await validator.stop()


def main(ports: Ports, run_dir: Path, genesis: float) -> int:
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
    chain = FakeChain(hotkeys, genesis, BLOCK_SECONDS, TEMPO)
    asyncio.run(serve(SandboxValidator(chain)))
    return 0
