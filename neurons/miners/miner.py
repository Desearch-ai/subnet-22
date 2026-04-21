import argparse
import asyncio
import copy
import os
import sys
import time
import traceback
from abc import ABC, abstractmethod
from collections import deque
from functools import partial
from typing import Dict, Tuple

import bittensor as bt
from bittensor.core.metagraph import AsyncMetagraph
from openai import OpenAI

import desearch
from desearch.miner_config import load_miner_manifest
from desearch.protocol import (
    IsAlive,
    ScraperStreamingSynapse,
    TwitterIDSearchSynapse,
    TwitterSearchSynapse,
    TwitterURLsSearchSynapse,
    WebSearchSynapse,
)
from neurons.miners.config import check_config, get_config
from neurons.miners.scraper_miner import ScraperMiner
from neurons.miners.twitter_search_miner import TwitterSearchMiner
from neurons.miners.web_search_miner import WebSearchMiner

OpenAI.api_key = os.environ.get("OPENAI_API_KEY")
if not OpenAI.api_key:
    raise ValueError(
        "Please set the OPENAI_API_KEY environment variable. See here: https://github.com/Desearch-ai/subnet-22/blob/main/docs/env_variables.md"
    )


class StreamMiner(ABC):
    subtensor: "bt.AsyncSubtensor"
    metagraph: "AsyncMetagraph"
    wallet: "bt.Wallet"
    axon: "bt.Axon"

    def __init__(self, config=None, wallet=None):
        bt.logging.info("starting stream miner")

        base_config = copy.deepcopy(config or get_config())
        self.config = self.config()
        self.config.merge(base_config)
        check_config(StreamMiner, self.config)
        bt.logging.info(self.config)

        self.prompt_cache: Dict[str, Tuple[str, int]] = {}
        self.request_timestamps: Dict = {}
        self.manifest = load_miner_manifest(self.config.miner.config_path)

        bt.logging(config=self.config, logging_dir=self.config.full_path)
        bt.logging.on()
        bt.logging.set_info(True)

        if self.config.logging.debug:
            bt.logging.set_debug(True)
        if self.config.logging.trace:
            bt.logging.set_trace(True)

        bt.logging.info("Setting up bittensor objects.")

        self.wallet = wallet or bt.Wallet(config=self.config)
        bt.logging.info(f"Wallet {self.wallet}")

        self.should_exit: bool = False
        self.my_subnet_uid: int | None = None
        self.last_epoch_block: int | None = None
        self.lock = asyncio.Lock()

    async def initialize(self):
        self.subtensor = bt.AsyncSubtensor(
            config=self.config, websocket_shutdown_timer=None
        )
        await self.subtensor.initialize()
        bt.logging.info(f"Subtensor: {self.subtensor}")
        bt.logging.info(
            f"Running miner for subnet: {self.config.netuid} on network: {self.subtensor.chain_endpoint}"
        )

        self.metagraph = await self.subtensor.metagraph(self.config.netuid)
        bt.logging.info(f"Metagraph: {self.metagraph}")

        if self.wallet.hotkey.ss58_address not in self.metagraph.hotkeys:
            bt.logging.error(
                f"\nYour miner: {self.wallet} is not registered to chain connection: {self.subtensor} \nRun btcli register and try again. "
            )
            sys.exit()

        self.my_subnet_uid = self.metagraph.hotkeys.index(
            self.wallet.hotkey.ss58_address
        )
        bt.logging.info(f"Running miner on uid: {self.my_subnet_uid}")

        if self.config.axon.external_ip is not None:
            bt.logging.debug(
                f"Starting axon on port {self.config.axon.port} and external ip {self.config.axon.external_ip}"
            )
            self.axon = bt.Axon(
                wallet=self.wallet,
                port=self.config.axon.port,
                external_ip=self.config.axon.external_ip,
            )
        else:
            bt.logging.debug(f"Starting axon on port {self.config.axon.port}")
            self.axon = bt.Axon(wallet=self.wallet, port=self.config.axon.port)

        bt.logging.info("Attaching forward function to axon.")

        self.axon.attach(
            forward_fn=self._is_alive,
            blacklist_fn=self.blacklist_is_alive,
        ).attach(
            forward_fn=self._smart_scraper,
            blacklist_fn=self.blacklist_smart_scraper,
        ).attach(
            forward_fn=self._twitter_search,
            blacklist_fn=self.blacklist_twitter_search,
        ).attach(
            forward_fn=self._twitter_id_search,
            blacklist_fn=self.blacklist_twitter_id_search,
        ).attach(
            forward_fn=self._twitter_urls_search,
            blacklist_fn=self.blacklist_twitter_urls_search,
        ).attach(
            forward_fn=self._web_search,
            blacklist_fn=self.blacklist_web_search,
        )

        bt.logging.info(f"Axon created: {self.axon}")

    @abstractmethod
    def config(self) -> "bt.Config": ...

    @classmethod
    @abstractmethod
    def add_args(cls, parser: argparse.ArgumentParser): ...

    async def _is_alive(self, synapse: IsAlive) -> IsAlive:
        bt.logging.info("answered to be active")

        try:
            self.manifest = load_miner_manifest(self.config.miner.config_path)
        except Exception as e:
            bt.logging.warning(
                f"Failed to reload miner manifest, using cached value: {e}"
            )

        synapse.manifest = self.manifest.model_dump()
        return synapse

    async def _smart_scraper(
        self, synapse: ScraperStreamingSynapse
    ) -> ScraperStreamingSynapse:
        return await self.smart_scraper(synapse)

    async def _twitter_search(
        self, synapse: TwitterSearchSynapse
    ) -> TwitterSearchSynapse:
        return await self.twitter_search(synapse)

    async def _twitter_id_search(
        self, synapse: TwitterIDSearchSynapse
    ) -> TwitterIDSearchSynapse:
        return await self.twitter_id_search(synapse)

    async def _twitter_urls_search(
        self, synapse: TwitterURLsSearchSynapse
    ) -> TwitterURLsSearchSynapse:
        return await self.twitter_urls_search(synapse)

    async def _web_search(self, synapse: WebSearchSynapse) -> WebSearchSynapse:
        return await self.web_search(synapse)

    async def base_blacklist(self, synapse) -> Tuple[bool, str]:
        try:
            hotkey = synapse.dendrite.hotkey
            synapse_type = type(synapse).__name__

            if hotkey in desearch.BLACKLISTED_KEYS:
                return True, f"Blacklisted a {synapse_type} request from {hotkey}"

            uid = None
            for _uid, _axon in enumerate(self.metagraph.axons):
                if _axon.hotkey == hotkey:
                    uid = _uid
                    break

            if uid is None:
                return (
                    True,
                    f"Blacklisted a non registered hotkey's {synapse_type} request from {hotkey}",
                )

            if self.config.subtensor.network == "finney":
                alpha_stake = float(self.metagraph.alpha_stake[uid].item())
                total_stake = float(self.metagraph.total_stake[uid].item())

                if (
                    alpha_stake < desearch.MIN_ALPHA_STAKE
                    or total_stake < desearch.MIN_TOTAL_STAKE
                ):
                    return (
                        True,
                        (
                            f"Blacklisted a low stake {synapse_type} request: "
                            f"alpha_stake={alpha_stake} < {desearch.MIN_ALPHA_STAKE} "
                            f"or total_stake={total_stake} < {desearch.MIN_TOTAL_STAKE} "
                            f"from {hotkey}"
                        ),
                    )

            time_window = desearch.MIN_REQUEST_PERIOD * 60
            current_time = time.time()

            async with self.lock:
                if hotkey not in self.request_timestamps:
                    self.request_timestamps[hotkey] = deque()

                while (
                    self.request_timestamps[hotkey]
                    and current_time - self.request_timestamps[hotkey][0] > time_window
                ):
                    self.request_timestamps[hotkey].popleft()

                if len(self.request_timestamps[hotkey]) >= desearch.MAX_REQUESTS:
                    return (
                        True,
                        f"Request frequency for {hotkey} exceeded: {len(self.request_timestamps[hotkey])} requests in {desearch.MIN_REQUEST_PERIOD} minutes. Limit is {desearch.MAX_REQUESTS} requests.",
                    )

                self.request_timestamps[hotkey].append(current_time)

            return False, f"accepting {synapse_type} request from {hotkey}"

        except Exception:
            bt.logging.error(f"error in blacklist {traceback.format_exc()}")
            return True, "error in blacklist"

    async def blacklist_is_alive(self, synapse: IsAlive) -> Tuple[bool, str]:
        blacklist = await self.base_blacklist(synapse)
        bt.logging.debug(blacklist[1])
        return blacklist

    async def blacklist_smart_scraper(
        self, synapse: ScraperStreamingSynapse
    ) -> Tuple[bool, str]:
        blacklist = await self.base_blacklist(synapse)
        bt.logging.info(blacklist[1])
        return blacklist

    async def blacklist_twitter_search(
        self, synapse: TwitterSearchSynapse
    ) -> Tuple[bool, str]:
        blacklist = await self.base_blacklist(synapse)
        bt.logging.info(blacklist[1])
        return blacklist

    async def blacklist_twitter_id_search(
        self, synapse: TwitterIDSearchSynapse
    ) -> Tuple[bool, str]:
        blacklist = await self.base_blacklist(synapse)
        bt.logging.info(blacklist[1])
        return blacklist

    async def blacklist_twitter_urls_search(
        self, synapse: TwitterURLsSearchSynapse
    ) -> Tuple[bool, str]:
        blacklist = await self.base_blacklist(synapse)
        bt.logging.info(blacklist[1])
        return blacklist

    async def blacklist_web_search(self, synapse: WebSearchSynapse) -> Tuple[bool, str]:
        blacklist = await self.base_blacklist(synapse)
        bt.logging.info(blacklist[1])
        return blacklist

    @abstractmethod
    async def smart_scraper(
        self, synapse: ScraperStreamingSynapse
    ) -> ScraperStreamingSynapse: ...

    @abstractmethod
    async def twitter_search(
        self, synapse: TwitterSearchSynapse
    ) -> TwitterSearchSynapse: ...

    @abstractmethod
    async def twitter_id_search(
        self, synapse: TwitterIDSearchSynapse
    ) -> TwitterIDSearchSynapse: ...

    @abstractmethod
    async def twitter_urls_search(
        self, synapse: TwitterURLsSearchSynapse
    ) -> TwitterURLsSearchSynapse: ...

    @abstractmethod
    async def web_search(self, synapse: WebSearchSynapse) -> WebSearchSynapse: ...

    async def sync_metagraph_loop(self):
        first_run = True

        while not self.should_exit:
            try:
                if first_run:
                    bt.logging.debug("Skipping first metagraph sync")
                    first_run = False
                else:
                    await self.metagraph.sync(subtensor=self.subtensor)
                    bt.logging.info("Resynced metagraph in background")

                await asyncio.sleep(900)
            except Exception as e:
                bt.logging.error(f"Error during metagraph sync: {e}")

                try:
                    self.subtensor = bt.AsyncSubtensor(
                        config=self.config, websocket_shutdown_timer=None
                    )
                    await self.subtensor.initialize()
                    self.metagraph = await self.subtensor.metagraph(self.config.netuid)
                except Exception as e:
                    bt.logging.error(
                        f"Error during metagraph sync - reconnection to subtensor also failed: {e}"
                    )

                bt.logging.info("Retrying in 2 minutes")
                await asyncio.sleep(120)

    async def run(self):
        if not await self.subtensor.is_hotkey_registered(
            netuid=self.config.netuid,
            hotkey_ss58=self.wallet.hotkey.ss58_address,
        ):
            bt.logging.error(
                f"Wallet: {self.wallet} is not registered on netuid {self.config.netuid}"
                f"Please register the hotkey using `btcli s register --netuid 18` before trying again"
            )
            sys.exit()

        bt.logging.info(
            f"Serving axon {ScraperStreamingSynapse} on network: {self.config.subtensor.chain_endpoint} with netuid: {self.config.netuid}"
        )
        await self.subtensor.serve_axon(axon=self.axon, netuid=self.config.netuid)

        bt.logging.info(f"Starting axon server on port: {self.config.axon.port}")
        self.axon.start()

        self.last_epoch_block = await self.subtensor.get_current_block()
        bt.logging.info(f"Miner starting at block: {self.last_epoch_block}")
        bt.logging.info("Starting main loop")

        sync_task = asyncio.create_task(self.sync_metagraph_loop())

        step = 0
        try:
            while not self.should_exit:
                current_block = await self.subtensor.get_current_block()

                while (
                    current_block - self.last_epoch_block
                    < self.config.miner.blocks_per_epoch
                ):
                    await asyncio.sleep(60)
                    current_block = await self.subtensor.get_current_block()

                    if self.should_exit:
                        break

                self.last_epoch_block = await self.subtensor.get_current_block()

                metagraph = await self.subtensor.metagraph(
                    netuid=self.config.netuid,
                    lite=True,
                    block=self.last_epoch_block,
                )

                log = (
                    f"Step:{step} | "
                    f"Block:{metagraph.block.item()} | "
                    f"Stake:{metagraph.S[self.my_subnet_uid]} | "
                    f"Consensus:{metagraph.C[self.my_subnet_uid]} | "
                    f"Incentive:{metagraph.I[self.my_subnet_uid]} | "
                    f"Emission:{metagraph.E[self.my_subnet_uid]}"
                )
                bt.logging.info(log)

                step += 1

        except asyncio.CancelledError:
            bt.logging.info("Miner run loop cancelled.")
            raise
        except Exception:
            bt.logging.error(traceback.format_exc())
        finally:
            self.should_exit = True
            sync_task.cancel()

            try:
                await sync_task
            except (asyncio.CancelledError, Exception):
                pass

    async def start(self):
        await self.initialize()
        await self.run()

    async def stop(self):
        bt.logging.info("Stopping miner.")
        self.should_exit = True

        if hasattr(self, "axon") and self.axon is not None:
            self.axon.stop()

        if hasattr(self, "subtensor"):
            await self.subtensor.close()


class StreamingTemplateMiner(StreamMiner):
    def config(self) -> "bt.Config":
        parser = argparse.ArgumentParser(description="Streaming Miner Configs")
        self.add_args(parser)
        return bt.Config(parser)

    def add_args(cls, parser: argparse.ArgumentParser):
        pass

    async def smart_scraper(
        self, synapse: ScraperStreamingSynapse
    ) -> ScraperStreamingSynapse:
        bt.logging.info(f"started processing for synapse {synapse}")
        tw_miner = ScraperMiner(self)
        token_streamer = partial(tw_miner.smart_scraper, synapse)
        return synapse.create_streaming_response(token_streamer)

    async def twitter_search(
        self, synapse: TwitterSearchSynapse
    ) -> TwitterSearchSynapse:
        bt.logging.info(f"started processing for twitter search synapse {synapse}")
        twitter_search_miner = TwitterSearchMiner(self)
        return await twitter_search_miner.search(synapse)

    async def twitter_id_search(
        self, synapse: TwitterIDSearchSynapse
    ) -> TwitterIDSearchSynapse:
        bt.logging.info(f"started processing for search ID synapse {synapse}")
        twitter_search_miner = TwitterSearchMiner(self)
        return await twitter_search_miner.search_by_id(synapse)

    async def twitter_urls_search(
        self, synapse: TwitterURLsSearchSynapse
    ) -> TwitterURLsSearchSynapse:
        bt.logging.info(f"started processing for search URL synapse {synapse}")
        twitter_search_miner = TwitterSearchMiner(self)
        return await twitter_search_miner.search_by_urls(synapse)

    async def web_search(self, synapse: WebSearchSynapse) -> WebSearchSynapse:
        bt.logging.info(f"started processing for Web search  synapse {synapse}")
        web_search_miner = WebSearchMiner(self)
        return await web_search_miner.search(synapse)


async def main():
    miner = StreamingTemplateMiner()

    try:
        await miner.start()
    except KeyboardInterrupt:
        bt.logging.success("Miner killed by keyboard interrupt.")
    finally:
        await miner.stop()


if __name__ == "__main__":
    asyncio.run(main())
