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
from desearch.embedding import MODELS, OPENROUTER_EMBEDDINGS, EmbeddingClient
from desearch.credit import SHARE_WINDOW_H
from desearch.fetch import Fetcher, ScrapingDog
from desearch.kinds import CRAWL, EMBED
from desearch.manifest import seed_from_hash
from bittensor.wallets import Wallet

from neurons.validators.config import ENV_FILE, config, flat, setup_logging
from neurons.validators.crawl import CrawlValidator
from neurons.validators.embed import EmbedValidator
from neurons.validators.fetchers import OWN_IP_SETTINGS, SampleFetcher
from neurons.validators.ledger import Ledger
from neurons.validators.upload_log import UploadLog
from neurons.validators.weights import set_weights, weights_from_shares

WEIGHTS_WINDOW_BLOCKS = 20
UPLOAD_LOG_POLL_S = 60
PRUNE_EVERY_S = 3600
METAGRAPH_SYNC_S = 600
POLL_S = 60
AFTER_WEIGHTS_S = 300
SHARES_TIMEOUT = aiohttp.ClientTimeout(total=30.0)
SIGNER_RETRY_S = 30
VALIDATION_JOBS = 12
EMBED_JOBS = 2
# The engine pins the same hosts for its queries; SiliconFlow is left out as it runs fp8.
EMBED_PROVIDERS = "DeepInfra,Nebius"
SCRAPINGDOG_CONCURRENCY = 50
SCRAPINGDOG_TIMEOUT_S = 30.0
DOWNLOAD_TIMEOUT = aiohttp.ClientTimeout(total=120.0)
WANDB_PROJECT = "smart-scrape-1.0"
WANDB_ENTITY = "smart-scrape"

log = logging.getLogger("validator")


class Validator:
    def __init__(self):
        env.load(ENV_FILE)
        self.config = config()
        setup_logging(self.config)
        log.info(str(flat(self.config)))
        self.scrapingdog_key = os.environ.get("SCRAPINGDOG_API_KEY", "")
        self.ledger = Ledger(os.path.join(self.config.neuron.full_path, "verdicts.db"))
        self.stopping = asyncio.Event()

    async def initialize(self) -> None:
        network = self.config.subtensor.chain_endpoint or self.config.subtensor.network
        log.info(f"Running validator for subnet {self.config.netuid} on {network}")
        self.wallet = Wallet(
            name=self.config.wallet.name,
            hotkey=self.config.wallet.hotkey,
            path=self.config.wallet.path,
        )
        self.subtensor = await bt.Subtensor(network)
        self.metagraph = await self.subtensor.subnets.metagraph(self.config.netuid)
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
            log.error(
                f"Not checking crawl tasks, so not setting weights either: {reason}"
            )
            await self.stopping.wait()
            return

        signer = await self.api_signer()
        async with (
            TaskApiClient(self.config.neuron.task_api_url, self.wallet.hotkey) as api,
            ScrapingDog(
                self.scrapingdog_key,
                SCRAPINGDOG_CONCURRENCY,
                timeout=SCRAPINGDOG_TIMEOUT_S,
            ) as scrapingdog,
        ):
            fetcher = SampleFetcher(Fetcher(OWN_IP_SETTINGS), scrapingdog)
            validator = self.crawl_checker = CrawlValidator(
                api,
                fetcher,
                self.http,
                ledger=self.ledger,
                storage_url=self.config.neuron.storage_url,
                seeds=self.seed_for,
                signer=signer,
            )
            log.info(
                f"Checking crawl uploads listed at {self.config.neuron.storage_url},"
                f" reporting to {self.config.neuron.task_api_url}"
            )
            following = asyncio.create_task(self.follow_upload_log(signer))
            try:
                await asyncio.gather(
                    *(validator.run(self.stopping) for _ in range(VALIDATION_JOBS))
                )
            finally:
                following.cancel()
                await fetcher.aclose()
        requests = scrapingdog.requests
        log.info(
            f"Crawl validation stopped: {fetcher.own_ip_fetches} samples fetched from"
            f" our own IP, {fetcher.scrapingdog_fetches} through ScrapingDog"
            f" ({requests['plain']} plain and {requests['rendered']} rendered requests)"
        )

    async def check_embed_tasks(self) -> None:
        key = os.environ.get("EMBED_API_KEY", "")
        if not key:
            log.info(
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
            name: EmbeddingClient(url, key, model, providers)
            for name, model in MODELS.items()
        }
        signer = await self.api_signer()
        async with TaskApiClient(
            self.config.neuron.task_api_url, self.wallet.hotkey
        ) as api:
            checker = EmbedValidator(
                api,
                self.http,
                references,
                ledger=self.ledger,
                storage_url=self.config.neuron.storage_url,
                seeds=self.seed_for,
                signer=signer,
            )
            log.info(f"Validating embed tasks through {url}")
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
            config=flat(self.config),
            dir=self.config.neuron.full_path,
            reinit="finish_previous",
        )
        self.config.signature = self.wallet.hotkey.sign(run.id.encode()).hex()
        wandb.config.update(flat(self.config), allow_val_change=True)
        log.info(f"Started wandb run for project '{WANDB_PROJECT}'")

    def crawl_disabled_reason(self) -> str | None:
        if not self.scrapingdog_key:
            return f"SCRAPINGDOG_API_KEY is not set, add it to {ENV_FILE}"
        if not self.config.neuron.storage_url:
            return (
                "--neuron.storage_url is not set: the public URL of the uploads bucket"
            )
        return None

    async def seed_for(self, block: int) -> str | None:
        """The sample seed for an upload, from the chain itself; None until its block exists."""
        if block > await self.subtensor.block():
            return None
        return seed_from_hash((await self.subtensor.block_info(block)).hash)

    async def api_signer(self) -> str:
        """The key the task API signs manifests with, fetched once."""
        url = f"{self.config.neuron.task_api_url.rstrip('/')}/v1/key"
        while not self.stopping.is_set():
            try:
                async with self.http.get(
                    url, timeout=SHARES_TIMEOUT, raise_for_status=True
                ) as response:
                    return (await response.json())["signer"]
            except Exception as error:
                log.warning(f"Task API signer unavailable, retrying: {error!r}")
                await asyncio.sleep(SIGNER_RETRY_S)
        return ""

    async def follow_upload_log(self, signer: str) -> None:
        """Every miner's uploads, from the log the task API signs into the uploads bucket."""
        upload_log = UploadLog(
            self.http, self.config.neuron.storage_url, signer, self.ledger
        )
        pruned_at = 0.0
        while not self.stopping.is_set():
            try:
                added = await upload_log.poll()
                if added:
                    log.info(f"{added} uploads read from the upload log")
            except Exception as error:
                log.warning(f"Upload log unavailable, retrying: {error!r}")
            if time.time() - pruned_at > PRUNE_EVERY_S:
                self.ledger.prune()
                pruned_at = time.time()
            await asyncio.sleep(UPLOAD_LOG_POLL_S)

    async def weights(self) -> np.ndarray | None:
        """Crawl from every miner's logged uploads judged by this validator's own checks, embed from its own checks."""
        paid = self.ledger.crawl_paid()
        crawl = proportions({miner: found.rows for miner, found in paid.items()})
        shares = {CRAWL: crawl} if crawl else {}
        if embed := self.ledger.shares().get(EMBED):
            shares[EMBED] = embed
        if not shares:
            log.warning(
                "No paid work in the scoring window yet; all weight goes to burn"
            )
        self.report_window(shares, paid)
        weights = weights_from_shares(list(self.metagraph.hotkeys), shares)
        return weights if weights.any() else None

    def report_window(self, shares: dict, paid: dict) -> None:
        """Each miner's share, its uploads, and what this validator's own checks of it found."""
        log.info(f"Scoring window, last {SHARE_WINDOW_H} h:")
        for miner, found in sorted(paid.items(), key=lambda item: -item[1].rows):
            share = shares.get(CRAWL, {}).get(miner, 0.0)
            log.info(
                f"  crawl {miner[:10]}: {found.uploads} uploads, checked {found.checked}"
                f" ({found.failed} failed), paid at {found.rate:.2f},"
                f" {found.rows:.0f} rows, share {share:.3f}"
            )
        for miner, share in sorted(shares.get(EMBED, {}).items(), key=lambda i: -i[1]):
            log.info(f"  embed {miner[:10]}: share {share:.3f}")

    async def sync_weights(self) -> None:
        while True:
            try:
                blocks_left = await self.blocks_until_next_epoch()
                log.info(f"Blocks left until next epoch: {blocks_left}")
                if blocks_left <= WEIGHTS_WINDOW_BLOCKS and self.should_set_weights():
                    started = time.time()
                    # Fresh, so a UID that changed hands since the last sync is not paid.
                    self.metagraph = await self.subtensor.subnets.metagraph(
                        self.config.netuid
                    )
                    weights = await self.weights()
                    if weights is not None:
                        await set_weights(self, weights)
                    log.info(f"Weight setting took {time.time() - started:.2f}s")
                    await asyncio.sleep(AFTER_WEIGHTS_S)
            except Exception as error:
                log.error(f"Error while setting weights: {error}")
            await asyncio.sleep(POLL_S)

    async def sync_metagraph(self) -> None:
        while True:
            await asyncio.sleep(METAGRAPH_SYNC_S)
            try:
                await self.check_registered()
                self.metagraph = await self.subtensor.subnets.metagraph(
                    self.config.netuid
                )
                log.info(f"Metagraph synced: {self.metagraph.num_uids} uids")
            except Exception as error:
                log.error(f"Error while syncing the metagraph: {error}")

    async def blocks_until_next_epoch(self) -> int:
        return await self.subtensor.epochs.blocks_until_next_epoch(
            netuid=self.config.netuid
        )

    async def check_registered(self) -> None:
        uid = await self.subtensor.neurons.uid(
            hotkey_ss58=self.wallet.hotkey.ss58_address, netuid=self.config.netuid
        )
        if uid is None:
            log.error(
                f"Hotkey {self.wallet.hotkey.ss58_address} is not registered on netuid {self.config.netuid}."
                " Please register the hotkey using `btcli subnets register` before trying again"
            )
            sys.exit()

    def should_set_weights(self) -> bool:
        if self.config.neuron.disable_set_weights:
            log.info("Weight setting is disabled by configuration.")
            return False
        if reason := self.checker_trouble():
            log.error(f"Not setting weights: {reason}")
            return False
        return True

    def checker_trouble(self) -> str | None:
        """A validator that cannot check tasks has no standing to weight them."""
        if reason := self.crawl_disabled_reason():
            return reason
        checker = getattr(self, "crawl_checker", None)
        return checker.trouble if checker is not None else None

    async def stop(self) -> None:
        log.info("Stopping validator")
        if hasattr(self, "http"):
            await self.http.close()
        if hasattr(self, "subtensor"):
            await self.subtensor.close()


def proportions(amounts: dict[str, float]) -> dict[str, float]:
    earned = {miner: amount for miner, amount in amounts.items() if amount > 0}
    total = sum(earned.values())
    return {miner: amount / total for miner, amount in earned.items()} if total else {}


async def main() -> None:
    validator = Validator()
    try:
        await validator.run()
    finally:
        await validator.stop()


if __name__ == "__main__":
    asyncio.run(main())
