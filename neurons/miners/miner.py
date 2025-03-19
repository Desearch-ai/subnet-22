import os
import time
import copy
import wandb
import json
import pathlib
import asyncio
import datura
import argparse
import threading
import traceback
import bittensor as bt

from openai import OpenAI
from functools import partial
from collections import deque
from openai import AsyncOpenAI
from abc import ABC, abstractmethod
from neurons.miners.config import get_config, check_config
from typing import Dict, Tuple

from datura.utils import get_version

from datura.protocol import (
    IsAlive,
    ScraperStreamingSynapse,
    TwitterSearchSynapse,
    WebSearchSynapse,
    TwitterURLsSearchSynapse,
    TwitterIDSearchSynapse,
)
from neurons.miners.scraper_miner import ScraperMiner
from neurons.miners.twitter_search_miner import TwitterSearchMiner
from neurons.miners.web_search_miner import WebSearchMiner

OpenAI.api_key = os.environ.get("OPENAI_API_KEY")
if not OpenAI.api_key:
    raise ValueError(
        "Please set the OPENAI_API_KEY environment variable. See here: https://github.com/Datura-ai/desearch/blob/main/docs/env_variables.md"
    )

TWITTER_BEARER_TOKEN = os.environ.get("TWITTER_BEARER_TOKEN")
if not TWITTER_BEARER_TOKEN:
    raise ValueError(
        "Please set the TWITTER_BEARER_TOKEN environment variable. See here: https://github.com/Datura-ai/desearch/blob/main/docs/env_variables.md"
    )

netrc_path = pathlib.Path.home() / ".netrc"
wandb_api_key = os.getenv("WANDB_API_KEY")

print("WANDB_API_KEY is set:", bool(wandb_api_key))
print("~/.netrc exists:", netrc_path.exists())

if not wandb_api_key and not netrc_path.exists():
    raise ValueError(
        "Please log in to wandb using `wandb login` or set the WANDB_API_KEY environment variable."
    )

client = AsyncOpenAI(timeout=60.0)
valid_hotkeys = []


class StreamMiner(ABC):
    def __init__(self, config=None, axon=None, wallet=None, subtensor=None):
        bt.logging.info("starting stream miner")
        base_config = copy.deepcopy(config or get_config())
        self.config = self.config()
        self.config.merge(base_config)
        check_config(StreamMiner, self.config)
        bt.logging.info(self.config)  # TODO: duplicate print?
        self.prompt_cache: Dict[str, Tuple[str, int]] = {}
        self.request_timestamps = {}

        # Activating Bittensor's logging with the set configurations.
        bt.logging(config=self.config, logging_dir=self.config.full_path)
        bt.logging.on()
        bt.logging.set_info(True)

        if self.config.logging.debug:
            bt.logging.set_debug(True)
        if self.config.logging.trace:
            bt.logging.set_trace(True)

        bt.logging.info("Setting up bittensor objects.")

        # Wallet holds cryptographic information, ensuring secure transactions and communication.
        self.wallet = wallet or bt.wallet(config=self.config)
        bt.logging.info(f"Wallet {self.wallet}")

        # subtensor manages the blockchain connection, facilitating interaction with the Bittensor blockchain.
        self.subtensor = subtensor or bt.subtensor(config=self.config)
        bt.logging.info(f"Subtensor: {self.subtensor}")
        bt.logging.info(
            f"Running miner for subnet: {self.config.netuid} on network: {self.subtensor.chain_endpoint} with config:"
        )

        # metagraph provides the network's current state, holding state about other participants in a subnet.
        self.metagraph = self.subtensor.metagraph(self.config.netuid)
        bt.logging.info(f"Metagraph: {self.metagraph}")

        if self.wallet.hotkey.ss58_address not in self.metagraph.hotkeys:
            bt.logging.error(
                f"\nYour validator: {self.wallet} if not registered to chain connection: {self.subtensor} \nRun btcli register and try again. "
            )
            exit()
        else:
            # Each miner gets a unique identity (UID) in the network for differentiation.
            self.my_subnet_uid = self.metagraph.hotkeys.index(
                self.wallet.hotkey.ss58_address
            )
            bt.logging.info(f"Running miner on uid: {self.my_subnet_uid}")

        # The axon handles request processing, allowing validators to send this process requests.
        if axon is not None:
            self.axon = axon
        elif self.config.axon.external_ip is not None:
            bt.logging.debug(
                f"Starting axon on port {self.config.axon.port} and external ip {self.config.axon.external_ip}"
            )
            self.axon = bt.axon(
                wallet=self.wallet,
                port=self.config.axon.port,
                external_ip=self.config.axon.external_ip,
            )
        else:
            bt.logging.debug(f"Starting axon on port {self.config.axon.port}")
            self.axon = bt.axon(wallet=self.wallet, port=self.config.axon.port)

        # Attach determiners which functions are called when servicing a request.
        bt.logging.info(f"Attaching forward function to axon.")
        print(f"Attaching forward function to axon. {self._is_alive}")

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
            forward_fn=self.twitter_urls_search,
            blacklist_fn=self.blacklist_twitter_urls_search,
        ).attach(
            forward_fn=self.web_search,
            blacklist_fn=self.blacklist_web_search,
        )

        bt.logging.info(f"Axon created: {self.axon}")

        # Instantiate runners
        self.should_exit: bool = False
        self.is_running: bool = False
        self.thread: threading.Thread = None
        self.lock = asyncio.Lock()
        self.request_timestamps: Dict = {}
        thread = threading.Thread(target=get_valid_hotkeys, args=(self.config,))
        # thread.start()

    @abstractmethod
    def config(self) -> "bt.Config": ...

    def _smart_scraper(
        self, synapse: ScraperStreamingSynapse
    ) -> ScraperStreamingSynapse:
        return self.smart_scraper(synapse)

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

    def base_blacklist(self, synapse, blacklist_amt=20000) -> Tuple[bool, str]:
        try:
            hotkey = synapse.dendrite.hotkey
            synapse_type = type(synapse).__name__

            if hotkey in datura.BLACKLISTED_KEYS:
                return True, f"Blacklisted a {synapse_type} request from {hotkey}"

            # if hotkey in datura.WHITELISTED_KEYS:
            #     return False, f"accepting {synapse_type} request from {hotkey}"

            # if hotkey not in datura.valid_validators:
            #     return (
            #         True,
            #         f"Blacklisted a {synapse_type} request from a non-valid hotkey: {hotkey}",
            #     )

            uid = None
            axon = None
            for _uid, _axon in enumerate(self.metagraph.axons):
                if _axon.hotkey == hotkey:
                    uid = _uid
                    axon = _axon
                    break

            if uid is None and datura.ALLOW_NON_REGISTERED == False:
                return (
                    True,
                    f"Blacklisted a non registered hotkey's {synapse_type} request from {hotkey}",
                )

            if self.config.subtensor.network == "finney":
                # check the stake
                tao = self.metagraph.neurons[uid].stake.tao
                # metagraph.neurons[uid].S
                if tao < blacklist_amt:
                    return (
                        True,
                        f"Blacklisted a low stake {synapse_type} request: {tao} < {blacklist_amt} from {hotkey}",
                    )

            time_window = datura.MIN_REQUEST_PERIOD * 60
            current_time = time.time()

            if hotkey not in self.request_timestamps:
                self.request_timestamps[hotkey] = deque()

            # Remove timestamps outside the current time window
            while (
                self.request_timestamps[hotkey]
                and current_time - self.request_timestamps[hotkey][0] > time_window
            ):
                self.request_timestamps[hotkey].popleft()

            # Check if the number of requests exceeds the limit
            if len(self.request_timestamps[hotkey]) >= datura.MAX_REQUESTS:
                return (
                    True,
                    f"Request frequency for {hotkey} exceeded: {len(self.request_timestamps[hotkey])} requests in {datura.MIN_REQUEST_PERIOD} minutes. Limit is {datura.MAX_REQUESTS} requests.",
                )

            self.request_timestamps[hotkey].append(current_time)

            return False, f"accepting {synapse_type} request from {hotkey}"

        except Exception as e:
            bt.logging.error(f"errror in blacklist {traceback.format_exc()}")

    def blacklist_is_alive(self, synapse: IsAlive) -> Tuple[bool, str]:
        blacklist = self.base_blacklist(synapse, datura.ISALIVE_BLACKLIST_STAKE)
        bt.logging.debug(blacklist[1])
        return blacklist

    def blacklist_smart_scraper(
        self, synapse: ScraperStreamingSynapse
    ) -> Tuple[bool, str]:
        blacklist = self.base_blacklist(
            synapse, datura.TWITTER_SCRAPPER_BLACKLIST_STAKE
        )
        bt.logging.info(blacklist[1])
        return blacklist

    def blacklist_twitter_search(
        self, synapse: TwitterSearchSynapse
    ) -> Tuple[bool, str]:
        blacklist = self.base_blacklist(
            synapse, datura.TWITTER_SCRAPPER_BLACKLIST_STAKE
        )
        bt.logging.info(blacklist[1])
        return blacklist

    def blacklist_twitter_id_search(
        self, synapse: TwitterIDSearchSynapse
    ) -> Tuple[bool, str]:
        blacklist = self.base_blacklist(
            synapse, datura.TWITTER_SCRAPPER_BLACKLIST_STAKE
        )
        bt.logging.info(blacklist[1])
        return blacklist

    def blacklist_twitter_urls_search(
        self, synapse: TwitterURLsSearchSynapse
    ) -> Tuple[bool, str]:
        blacklist = self.base_blacklist(
            synapse, datura.TWITTER_SCRAPPER_BLACKLIST_STAKE
        )
        bt.logging.info(blacklist[1])
        return blacklist

    def blacklist_web_search(self, synapse: WebSearchSynapse) -> Tuple[bool, str]:
        blacklist = self.base_blacklist(
            synapse, datura.TWITTER_SCRAPPER_BLACKLIST_STAKE
        )
        bt.logging.info(blacklist[1])
        return blacklist

    @classmethod
    @abstractmethod
    def add_args(cls, parser: argparse.ArgumentParser): ...

    async def _smart_scraper(
        self, synapse: ScraperStreamingSynapse
    ) -> ScraperStreamingSynapse:
        return self.smart_scraper(synapse)

    def _is_alive(self, synapse: IsAlive) -> IsAlive:
        bt.logging.info("answered to be active")
        synapse.completion = "True"
        return synapse

    @abstractmethod
    def smart_scraper(
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

    def sync_metagraph_with_interval(self):
        first_run = True

        while True:
            try:
                if first_run:
                    bt.logging.debug("Skipping first metagraph sync")
                    first_run = False
                else:
                    self.metagraph.sync(subtensor=self.subtensor)
                    bt.logging.info("Resynced metagraph in background")
                time.sleep(900)
            except Exception as e:
                bt.logging.error(f"Error during metagraph sync: {e}")

                try:
                    self.subtensor = bt.subtensor(config=self.config)
                    self.metagraph = self.subtensor.metagraph(self.config.netuid)
                except Exception as e:
                    bt.logging.error(
                        f"Error during metagraph sync - reconnection to subtensor also failed: {e}"
                    )

                bt.logging.info("Retrying in 2 minutes")
                time.sleep(120)

    def start_background_sync(self):
        self.sync_thread = threading.Thread(
            target=self.sync_metagraph_with_interval, daemon=True
        )
        self.sync_thread.start()

    def run(self):
        if not self.subtensor.is_hotkey_registered(
            netuid=self.config.netuid,
            hotkey_ss58=self.wallet.hotkey.ss58_address,
        ):
            bt.logging.error(
                f"Wallet: {self.wallet} is not registered on netuid {self.config.netuid}"
                f"Please register the hotkey using `btcli s register --netuid 18` before trying again"
            )
            exit()
        bt.logging.info(
            f"Serving axon {ScraperStreamingSynapse} on network: {self.config.subtensor.chain_endpoint} with netuid: {self.config.netuid}"
        )
        self.subtensor.serve_axon(axon=self.axon, netuid=self.config.netuid)
        bt.logging.info(f"Starting axon server on port: {self.config.axon.port}")
        self.axon.start()
        self.last_epoch_block = self.subtensor.get_current_block()
        bt.logging.info(f"Miner starting at block: {self.last_epoch_block}")
        bt.logging.info(f"Starting main loop")

        self.start_background_sync()

        step = 0
        try:
            while not self.should_exit:
                start_epoch = time.time()

                # --- Wait until next epoch.
                current_block = self.subtensor.get_current_block()
                while (
                    current_block - self.last_epoch_block
                    < self.config.miner.blocks_per_epoch
                ):
                    # --- Wait for next bloc.
                    time.sleep(60)
                    current_block = self.subtensor.get_current_block()
                    # --- Check if we should exit.
                    if self.should_exit:
                        break

                # --- Update the metagraph with the latest network state.
                self.last_epoch_block = self.subtensor.get_current_block()

                metagraph = self.subtensor.metagraph(
                    netuid=self.config.netuid,
                    lite=True,
                    block=self.last_epoch_block,
                )
                log = (
                    f"Step:{step} | "
                    f"Block:{metagraph.block.item()} | "
                    f"Stake:{metagraph.S[self.my_subnet_uid]} | "
                    f"Rank:{metagraph.R[self.my_subnet_uid]} | "
                    f"Trust:{metagraph.T[self.my_subnet_uid]} | "
                    f"Consensus:{metagraph.C[self.my_subnet_uid] } | "
                    f"Incentive:{metagraph.I[self.my_subnet_uid]} | "
                    f"Emission:{metagraph.E[self.my_subnet_uid]}"
                )
                bt.logging.info(log)

                step += 1

        except KeyboardInterrupt:
            self.axon.stop()
            bt.logging.success("Miner killed by keyboard interrupt.")
            exit()

        except Exception as e:
            bt.logging.error(traceback.format_exc())

    def run_in_background_thread(self):
        if not self.is_running:
            bt.logging.debug("Starting miner in background thread.")
            self.should_exit = False
            self.thread = threading.Thread(target=self.run, daemon=True)
            self.thread.start()
            self.is_running = True
            bt.logging.debug("Started")

    def stop_run_thread(self):
        if self.is_running:
            bt.logging.debug("Stopping miner in background thread.")
            self.should_exit = True
            self.sync_thread.join(5)
            self.thread.join(5)
            self.is_running = False
            bt.logging.debug("Stopped")

    def __enter__(self):
        self.run_in_background_thread()

    def __exit__(self, exc_type, exc_value, traceback):
        self.stop_run_thread()


class StreamingTemplateMiner(StreamMiner):
    def config(self) -> "bt.Config":
        parser = argparse.ArgumentParser(description="Streaming Miner Configs")
        self.add_args(parser)
        return bt.config(parser)

    def add_args(cls, parser: argparse.ArgumentParser):
        pass

    def smart_scraper(
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


def get_valid_hotkeys(config):
    global valid_hotkeys
    api = wandb.Api()
    subtensor = bt.subtensor(config=config)
    while True:
        metagraph = subtensor.metagraph(config.netuid)
        try:
            runs = api.runs(f"{datura.ENTITY}/{datura.PROJECT_NAME}")
            latest_version = get_version()
            for run in runs:
                if run.state == "running":
                    try:
                        # Extract hotkey and signature from the run's configuration
                        hotkey = run.config["hotkey"]
                        signature = run.config["signature"]
                        version = run.config["version"]
                        bt.logging.debug(
                            f"found running run of hotkey {hotkey}, {version} "
                        )

                        if latest_version == None:
                            bt.logging.error(f"Github API call failed!")
                            continue

                        if version != latest_version and latest_version != None:
                            bt.logging.debug(
                                f"Version Mismatch: Run version {version} does not match GitHub version {latest_version}"
                            )
                            continue

                        # Check if the hotkey is registered in the metagraph
                        if hotkey not in metagraph.hotkeys:
                            bt.logging.debug(
                                f"Invalid running run: The hotkey: {hotkey} is not in the metagraph."
                            )
                            continue

                        # Verify the signature using the hotkey
                        if not bt.Keypair(ss58_address=hotkey).verify(
                            run.id, bytes.fromhex(signature)
                        ):
                            bt.logging.debug(
                                f"Failed Signature: The signature: {signature} is not valid"
                            )
                            continue

                        if hotkey not in valid_hotkeys:
                            valid_hotkeys.append(hotkey)
                    except Exception as e:
                        bt.logging.debug(
                            f"exception in get_valid_hotkeys: {traceback.format_exc()}"
                        )

            bt.logging.info(f"total valid hotkeys list = {valid_hotkeys}")
            time.sleep(180)

        except json.JSONDecodeError as e:
            bt.logging.debug(f"JSON decoding error: {e} {run.id}")


if __name__ == "__main__":
    with StreamingTemplateMiner():
        while True:
            time.sleep(1)
