import asyncio
import logging
import os
import signal
import sys
import time

import aiohttp
import bittensor as bt
import numpy as np
import wandb

import desearch
from desearch import env
from desearch.client import TaskApiClient
from desearch.embedding import MODELS, OPENROUTER_EMBEDDINGS, HostedEmbedder
from desearch.fetch import Fetcher, ScrapingDog
from neurons.validators.config import ENV_FILE, add_args, check_config, config
from neurons.validators.crawl import CrawlValidator
from neurons.validators.embed import EmbedValidator
from neurons.validators.fetchers import OWN_IP_SETTINGS, SampleFetcher
from neurons.validators.weights import set_weights, weights_from_shares

WEIGHTS_WINDOW_BLOCKS = 20
METAGRAPH_SYNC_S = 600
POLL_S = 60
AFTER_WEIGHTS_S = 300
SHARES_TIMEOUT = aiohttp.ClientTimeout(total=30.0)
VALIDATION_JOBS = 4
EMBED_JOBS = 2
# The engine pins the same hosts for its queries; SiliconFlow is left out as it runs fp8.
EMBED_PROVIDERS = "DeepInfra,Nebius"
SCRAPINGDOG_CONCURRENCY = 8
DOWNLOAD_TIMEOUT = aiohttp.ClientTimeout(total=120.0)
WANDB_PROJECT = "smart-scrape-1.0"
WANDB_ENTITY = "smart-scrape"


class Validator:
    @classmethod
    def add_args(cls, parser):
        add_args(cls, parser)

    def __init__(self):
        env.load(ENV_FILE)
        self.config = config(Validator)
        check_config(self.config)
        bt.logging(config=self.config, logging_dir=self.config.neuron.full_path)
        bt.logging.set_config(self.config)
        bt.logging.register_primary_logger("validator")
        logging.getLogger("trafilatura").setLevel(logging.ERROR)
        bt.logging.info(str(self.config))
        self.scrapingdog_key = os.environ.get("SCRAPINGDOG_API_KEY", "")
        self.stopping = asyncio.Event()

    async def initialize(self) -> None:
        bt.logging.info(
            f"Running validator for subnet {self.config.netuid} on {self.config.subtensor.chain_endpoint}"
        )
        self.wallet = bt.Wallet(config=self.config)
        self.subtensor = bt.AsyncSubtensor(
            config=self.config, websocket_shutdown_timer=None
        )
        await self.subtensor.initialize()

        self.metagraph = await self.subtensor.metagraph(self.config.netuid)
        self.uid = self.metagraph.hotkeys.index(self.wallet.hotkey.ss58_address)
        self.http = aiohttp.ClientSession(timeout=DOWNLOAD_TIMEOUT)

    async def run(self) -> None:
        await self.initialize()
        self.init_wandb()
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(sig, self.stopping.set)

        background = [
            asyncio.create_task(self.sync_metagraph()),
            asyncio.create_task(self.sync_weights()),
        ]
        try:
            await asyncio.gather(self.check_crawl_tasks(), self.check_embed_tasks())
        finally:
            for task in background:
                task.cancel()
            await asyncio.gather(*background, return_exceptions=True)

    async def check_crawl_tasks(self) -> None:
        if reason := self.crawl_disabled_reason():
            bt.logging.error(
                f"Not checking crawl tasks, only setting weights: {reason}"
            )
            await self.stopping.wait()
            return

        async with (
            TaskApiClient(self.config.neuron.task_api_url, self.wallet.hotkey) as api,
            ScrapingDog(self.scrapingdog_key, SCRAPINGDOG_CONCURRENCY) as scrapingdog,
        ):
            fetcher = SampleFetcher(Fetcher(OWN_IP_SETTINGS), scrapingdog)
            validator = CrawlValidator(api, fetcher, self.http)
            bt.logging.info(
                f"Validating crawl tasks from {self.config.neuron.task_api_url}"
            )
            try:
                await asyncio.gather(
                    *(validator.run(self.stopping) for _ in range(VALIDATION_JOBS))
                )
            finally:
                await fetcher.aclose()
        requests = scrapingdog.requests
        bt.logging.info(
            f"Crawl validation stopped: {fetcher.own_ip_fetches} samples fetched from"
            f" our own IP, {fetcher.scrapingdog_fetches} through ScrapingDog"
            f" ({requests['plain']} plain and {requests['rendered']} rendered requests)"
        )

    async def check_embed_tasks(self) -> None:
        key = os.environ.get("EMBED_API_KEY", "")
        if not key:
            bt.logging.warning(
                f"Not checking embed tasks: EMBED_API_KEY is not set in {ENV_FILE}"
            )
            await self.stopping.wait()
            return

        url = os.environ.get("EMBED_API_URL", OPENROUTER_EMBEDDINGS)
        providers = tuple(
            p.strip()
            for p in os.environ.get("EMBED_PROVIDERS", EMBED_PROVIDERS).split(",")
            if p.strip()
        )
        references = {
            name: HostedEmbedder(url, key, model, providers)
            for name, model in MODELS.items()
        }
        async with TaskApiClient(
            self.config.neuron.task_api_url, self.wallet.hotkey
        ) as api:
            checker = EmbedValidator(api, self.http, references)
            bt.logging.info(f"Validating embed tasks through {url}")
            try:
                await asyncio.gather(
                    *(checker.run(self.stopping) for _ in range(EMBED_JOBS))
                )
            finally:
                for reference in references.values():
                    await reference.aclose()

    def init_wandb(self) -> None:
        if not self.config.wandb_on:
            return
        run_name = f"validator-{self.uid}-{desearch.__version__}"
        self.config.uid = self.uid
        self.config.hotkey = self.wallet.hotkey.ss58_address
        self.config.run_name = run_name
        self.config.version = desearch.__version__
        self.config.type = "validator"

        run = wandb.init(
            name=run_name,
            project=WANDB_PROJECT,
            entity=WANDB_ENTITY,
            config=self.config,
            dir=self.config.neuron.full_path,
            reinit="finish_previous",
        )
        self.config.signature = self.wallet.hotkey.sign(run.id.encode()).hex()
        wandb.config.update(self.config, allow_val_change=True)
        bt.logging.success(f"Started wandb run for project '{WANDB_PROJECT}'")

    def crawl_disabled_reason(self) -> str | None:
        if not self.scrapingdog_key:
            return f"SCRAPINGDOG_API_KEY is not set, add it to {ENV_FILE}"
        return None

    async def shares(self) -> dict[str, dict[str, float]]:
        url = f"{self.config.neuron.task_api_url.rstrip('/')}/v1/shares"
        async with self.http.get(
            url, timeout=SHARES_TIMEOUT, raise_for_status=True
        ) as response:
            pools = (await response.json()).get("pools", {})
        return {
            pool: {hotkey: float(share) for hotkey, share in shares.items()}
            for pool, shares in pools.items()
        }

    async def weights(self) -> np.ndarray | None:
        """None keeps the last weights when the task API is unreachable."""
        try:
            shares = await self.shares()
        except Exception as error:
            bt.logging.error(
                f"Crawl shares unavailable, keeping the last weights: {error!r}"
            )
            return None
        weights = weights_from_shares(list(self.metagraph.hotkeys), shares)
        return weights if weights.any() else None

    async def sync_weights(self) -> None:
        while True:
            try:
                blocks_left = await self.blocks_until_next_epoch()
                bt.logging.info(f"Blocks left until next epoch: {blocks_left}")
                if blocks_left <= WEIGHTS_WINDOW_BLOCKS and self.should_set_weights():
                    started = time.time()
                    weights = await self.weights()
                    if weights is not None:
                        await set_weights(self, weights)
                    bt.logging.info(f"Weight setting took {time.time() - started:.2f}s")
                    await asyncio.sleep(AFTER_WEIGHTS_S)
            except Exception as error:
                bt.logging.error(f"Error while setting weights: {error}")
            await asyncio.sleep(POLL_S)

    async def sync_metagraph(self) -> None:
        while True:
            await asyncio.sleep(METAGRAPH_SYNC_S)
            try:
                await self.check_registered()
                self.metagraph = await self.subtensor.metagraph(self.config.netuid)
                bt.logging.info(f"Metagraph synced: {int(self.metagraph.n)} uids")
            except Exception as error:
                bt.logging.error(f"Error while syncing the metagraph: {error}")

    async def blocks_until_next_epoch(self) -> int:
        current_block = await self.subtensor.get_current_block()
        tempo = await self.subtensor.tempo(self.config.netuid, current_block)
        return tempo - (current_block + self.config.netuid + 1) % (tempo + 1)

    async def check_registered(self) -> None:
        if not await self.subtensor.is_hotkey_registered(
            netuid=self.config.netuid,
            hotkey_ss58=self.wallet.hotkey.ss58_address,
        ):
            bt.logging.error(
                f"Wallet: {self.wallet} is not registered on netuid {self.config.netuid}."
                " Please register the hotkey using `btcli subnets register` before trying again"
            )
            sys.exit()

    def should_set_weights(self) -> bool:
        if self.config.neuron.disable_set_weights:
            bt.logging.info("Weight setting is disabled by configuration.")
            return False
        return True

    async def stop(self) -> None:
        bt.logging.info("Stopping validator")
        if hasattr(self, "http"):
            await self.http.close()
        if hasattr(self, "subtensor"):
            await self.subtensor.close()


async def main() -> None:
    validator = Validator()
    try:
        await validator.run()
    finally:
        await validator.stop()


if __name__ == "__main__":
    asyncio.run(main())
