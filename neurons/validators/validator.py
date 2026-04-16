import asyncio
import itertools
import sys
import time
from traceback import print_exception
from typing import Optional, Tuple

import bittensor as bt
import torch
from bittensor.core.metagraph import AsyncMetagraph

from desearch.protocol import IsAlive
from desearch.redis.redis_client import close_redis, initialize_redis
from desearch.redis.utils import (
    load_moving_averaged_scores,
    save_moving_averaged_scores,
)
from desearch.utils import resync_metagraph
from neurons.validators.base_validator import AbstractNeuron
from neurons.validators.clients.utility_api_client import UtilityAPIClient
from neurons.validators.config import add_args, check_config, config
from neurons.validators.proxy.uid_manager import UIDManager
from neurons.validators.scrapers.advanced_scraper_validator import AdvancedScraperValidator
from neurons.validators.scrapers.web_scraper_validator import WebScraperValidator
from neurons.validators.scrapers.x_scraper_validator import XScraperValidator
from neurons.validators.scoring.query_scheduler import QueryScheduler
from neurons.validators.scoring.scoring_store import ScoringStore
from neurons.validators.scoring.synthetic_query_generator import SyntheticQueryGenerator
from neurons.validators.scoring.weights import init_wandb, set_weights


class Neuron(AbstractNeuron):
    @classmethod
    def add_args(cls, parser):
        add_args(cls, parser)

    @classmethod
    def config(cls):
        return config(cls)

    subtensor: "bt.AsyncSubtensor"
    wallet: "bt.Wallet"
    metagraph: "AsyncMetagraph"

    loop: asyncio.AbstractEventLoop

    advanced_scraper_validator: "AdvancedScraperValidator"
    x_scraper_validator: "XScraperValidator"
    web_scraper_validator: "WebScraperValidator"

    moving_average_scores: torch.Tensor = None
    uid: int = None
    utility_api: UtilityAPIClient
    validator_identity: dict | None = None

    uid_manager: UIDManager

    def __init__(self):
        self.config = Neuron.config()
        check_config(self.config)
        bt.logging(config=self.config, logging_dir=self.config.neuron.full_path)
        bt.logging.set_config(self.config)
        print(self.config)
        bt.logging.info("neuron.__init__()")

        self.advanced_scraper_validator = AdvancedScraperValidator(neuron=self)
        self.x_scraper_validator = XScraperValidator(neuron=self)
        self.web_scraper_validator = WebScraperValidator(neuron=self)

        self.available_uids = []
        self.uid_manager = UIDManager()
        self.validator_identity = None

    async def initialize(self):
        bt.logging.info(
            f"Running validator for subnet: {self.config.netuid} on network: {self.config.subtensor.chain_endpoint}"
        )

        if self.config.neuron.offline:
            from desearch.bittensor.dendrite import Dendrite
            from desearch.bittensor.subtensor import Subtensor
            from desearch.bittensor.wallet import Wallet

            self.wallet = Wallet(config=self.config)
            self.subtensor = Subtensor(config=self.config)
            await self.subtensor.initialize()
            self.metagraph = await self.subtensor.metagraph(self.config.netuid)
            self.hotkeys = list(self.metagraph.hotkeys)

            self.dendrite_list = [
                Dendrite(wallet=self.wallet),
                Dendrite(wallet=self.wallet),
                Dendrite(wallet=self.wallet),
            ]
        else:
            self.wallet = bt.Wallet(config=self.config)

            self.subtensor = bt.AsyncSubtensor(
                config=self.config, websocket_shutdown_timer=None
            )
            await self.subtensor.initialize()

            self.metagraph = await self.subtensor.metagraph(self.config.netuid)

            self.hotkeys = list(self.metagraph.hotkeys)

            self.dendrite_list = [
                bt.Dendrite(wallet=self.wallet),
                bt.Dendrite(wallet=self.wallet),
                bt.Dendrite(wallet=self.wallet),
            ]

        self.dendrites = itertools.cycle(self.dendrite_list)

        self.uid = self.metagraph.hotkeys.index(self.wallet.hotkey.ss58_address)
        self.validator_identity = self._build_validator_identity()

        await initialize_redis()

    def _build_validator_identity(self) -> dict:
        hotkey = self.wallet.hotkey.ss58_address
        coldkey = next(
            (nr.coldkey for nr in self.metagraph.neurons if nr.hotkey == hotkey),
            None,
        )

        identity = {
            "uid": self.uid,
            "hotkey": hotkey,
            "coldkey": coldkey,
            "netuid": self.config.netuid,
        }

        external_ip = getattr(self.config.axon, "external_ip", None)
        port = getattr(self.config.axon, "port", None)

        if external_ip:
            identity["ip"] = external_ip
        if port:
            identity["port"] = port

        return identity

    async def sync_available_uids(self):
        start_time = time.time()

        try:
            self.available_uids = await self.get_available_uids_is_alive()

            self.uid_manager.resync(
                available_uids=self.available_uids, metagraph=self.metagraph
            )
        except Exception as e:
            bt.logging.error(
                f"sync_available_uids Failed to update available UIDs: {e}"
            )

        end_time = time.time()
        execution_time = end_time - start_time
        bt.logging.info(f"sync_available_uids finished in: {execution_time}s")

    async def check_uid(self, axon, uid):
        """Asynchronously check if a UID is available."""

        dendrite = next(self.dendrites)
        response = await dendrite(axon, IsAlive(), deserialize=False, timeout=10)

        if response.is_success:
            return axon
        else:
            raise Exception(f"UID {uid} is not active")

    async def get_available_uids_is_alive(self):
        """Get a dictionary of available UIDs and their axons asynchronously."""

        tasks = {
            uid.item(): self.check_uid(
                self.metagraph.axons[uid.item()],
                uid.item(),
            )
            for uid in self.metagraph.uids
        }

        results = await asyncio.gather(*tasks.values(), return_exceptions=True)

        available_uids = []
        unavailable_uids = []

        for uid, result in zip(tasks.keys(), results):
            if not isinstance(result, Exception):
                available_uids.append(uid)
            else:
                unavailable_uids.append(uid)

        bt.logging.info(
            f"Available UIDs: {available_uids}, total: {len(available_uids)}"
        )
        bt.logging.info(
            f"Unavailable UIDs: {unavailable_uids}, total: {len(unavailable_uids)}"
        )

        return available_uids

    async def get_random_miner(
        self, uid: Optional[int] = None
    ) -> Tuple[int, bt.AxonInfo]:
        """Return (uid, axon) for the given uid, or a random miner if uid is None."""
        if uid is not None:
            bt.logging.info(f"Run specific UID: {uid}")
            return uid, self.metagraph.axons[uid]

        selected_uid = self.uid_manager.get_miner_uid()

        bt.logging.info(f"Run random UID: {selected_uid}")

        return selected_uid, self.metagraph.axons[selected_uid]

    async def update_moving_averaged_scores(self, uids, rewards):
        try:
            # Ensure uids is a tensor
            if not isinstance(uids, torch.Tensor):
                uids = torch.tensor(
                    uids, dtype=torch.long, device=self.config.neuron.device
                )

            # Ensure rewards is also a tensor and on the correct device
            if not isinstance(rewards, torch.Tensor):
                rewards = torch.tensor(rewards, device=self.config.neuron.device)

            empty_rewards = torch.zeros(self.moving_averaged_scores.size()).to(
                self.config.neuron.device
            )

            scattered_rewards = empty_rewards.scatter(0, uids, rewards).to(
                self.config.neuron.device
            )

            average_reward = torch.mean(scattered_rewards)
            bt.logging.info(
                f"Scattered reward: {average_reward:.6f}"
            )  # Rounds to 6 decimal places for logging

            alpha = 0.2

            self.moving_averaged_scores = alpha * scattered_rewards + (
                1 - alpha
            ) * self.moving_averaged_scores.to(self.config.neuron.device)

            await save_moving_averaged_scores(self.moving_averaged_scores)

            bt.logging.info(
                f"Moving averaged scores: {torch.mean(self.moving_averaged_scores):.6f}"
            )  # Rounds to 6 decimal places for logging
            return scattered_rewards
        except Exception as e:
            bt.logging.error(f"Error in update_moving_averaged_scores: {e}")
            raise e

    async def blocks_until_next_epoch(self):
        bt.logging.info("Calculating block until next epoch")

        current_block = await self.subtensor.get_current_block()
        tempo = await self.subtensor.tempo(self.config.netuid, current_block)
        return tempo - (current_block + self.config.netuid + 1) % (tempo + 1)

    async def sync_metagraph(self):
        while True:
            try:
                await asyncio.sleep(10 * 60)  # 10 minutes

                bt.logging.info("Syncing metagraph and available UIDs")

                sync_start_time = time.time()

                # Ensure validator hotkey is still registered on the network.
                await self.check_registered()

                await resync_metagraph(self)
                await self.sync_available_uids()

                bt.logging.info(
                    f"Completed syncing metagraph and available UIDs: {time.time() - sync_start_time:.2f} seconds"
                )

            except Exception as e:
                bt.logging.error(f"Error in sync_metagraph: {e}")

    async def sync(self):
        """
        Weight-setting loop. Runs set_weights when within the last 20 blocks
        of an epoch.
        """

        while True:
            try:
                blocks_left = await self.blocks_until_next_epoch()

                bt.logging.info(f"Blocks left until next epoch: {blocks_left}")

                if blocks_left <= 20 and self.should_set_weights():
                    weight_set_start_time = time.time()
                    bt.logging.info("Setting weights as per condition.")
                    await set_weights(self)
                    weight_set_end_time = time.time()
                    bt.logging.info(
                        f"Weight setting execution time: {weight_set_end_time - weight_set_start_time:.2f} seconds"
                    )
                    await asyncio.sleep(300)

            except Exception as e:
                bt.logging.error(f"Error in validator sync: {e}")

            await asyncio.sleep(60)

    async def check_registered(self):
        if not await self.subtensor.is_hotkey_registered(
            netuid=self.config.netuid,
            hotkey_ss58=self.wallet.hotkey.ss58_address,
        ):
            bt.logging.error(
                f"Wallet: {self.wallet} is not registered on netuid {self.config.netuid}."
                f" Please register the hotkey using `btcli subnets register` before trying again"
            )
            sys.exit()

    def should_set_weights(self) -> bool:
        if self.config.neuron.disable_set_weights:
            bt.logging.info("Weight setting is disabled by configuration.")
            return False

        return True

    async def start(self):
        try:
            bt.logging.info("Starting Neuron")

            await self.initialize()
            await self.sync_available_uids()  # Initial sync

            self.loop = asyncio.get_event_loop()

            init_wandb(self)

            # Init Weights.
            bt.logging.debug("loading", "moving_averaged_scores")
            self.moving_averaged_scores = await load_moving_averaged_scores(
                self.metagraph, self.config
            )
            bt.logging.debug(str(self.moving_averaged_scores))

            scoring_store = ScoringStore()

            utility_api = UtilityAPIClient(
                base_url=self.config.neuron.utility_api_url,
                wallet=self.wallet,
            )
            self.utility_api = utility_api

            generator = SyntheticQueryGenerator()

            validators = {
                "ai_search": self.advanced_scraper_validator,
                "x_search": self.x_scraper_validator,
                "web_search": self.web_scraper_validator,
            }

            query_scheduler = QueryScheduler(
                neuron=self,
                generator=generator,
                scoring_store=scoring_store,
                validators=validators,
            )

            self.loop.create_task(self.sync_metagraph())
            self.loop.create_task(self.sync())
            self.loop.create_task(query_scheduler.run())

        except KeyboardInterrupt:
            self.axon.stop()
            bt.logging.success("Validator killed by keyboard interrupt.")
            sys.exit()
        except Exception as err:
            # In case of unforeseen errors, the validator will log the error and quit
            bt.logging.error("Error during validation", str(err))
            bt.logging.debug(print_exception(type(err), err, err.__traceback__))
            self.should_exit = True

    async def stop(self):
        bt.logging.info("Stopping Neuron")

        await close_redis()

        if hasattr(self, "utility_api"):
            await self.utility_api.close()

        await self.subtensor.close()

        for dendrite in self.dendrite_list:
            await dendrite.aclose_session()
