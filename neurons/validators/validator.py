import asyncio
import itertools
import random
import sys
import time
import traceback
from traceback import print_exception
from typing import Optional, Tuple

import bittensor as bt
import torch
from bittensor.core.metagraph import AsyncMetagraph

import wandb
from desearch import QUERY_MINERS
from desearch.protocol import IsAlive
from desearch.redis.redis_client import close_redis, initialize_redis
from desearch.redis.utils import (
    load_moving_averaged_scores,
    save_moving_averaged_scores,
)
from desearch.utils import (
    resync_metagraph,
    save_logs_in_chunks,
)
from neurons.validators.advanced_scraper_validator import AdvancedScraperValidator
from neurons.validators.base_validator import AbstractNeuron
from neurons.validators.basic_scraper_validator import BasicScraperValidator
from neurons.validators.basic_web_scraper_validator import BasicWebScraperValidator
from neurons.validators.config import add_args, check_config, config
from neurons.validators.proxy.uid_manager import UIDManager
from neurons.validators.synthetic_query_runner import SyntheticQueryRunnerMixin
from neurons.validators.validator_service_client import ValidatorServiceClient
from neurons.validators.weights import get_weights, init_wandb, set_weights


class Neuron(SyntheticQueryRunnerMixin, AbstractNeuron):
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
    x_scraper_validator: "BasicScraperValidator"

    moving_average_scores: torch.Tensor = None
    uid: int = None

    uid_manager: UIDManager

    def __init__(self):
        self.config = Neuron.config()
        check_config(self.config)
        bt.logging(config=self.config, logging_dir=self.config.neuron.full_path)
        bt.logging.set_config(self.config)
        print(self.config)
        bt.logging.info("neuron.__init__()")

        self.advanced_scraper_validator = AdvancedScraperValidator(neuron=self)
        self.x_scraper_validator = BasicScraperValidator(neuron=self)

        bt.logging.info("initialized_validators")

        self.organic_responses_computed = False
        self.available_uids = []

    async def initialize_components(self):
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
            self.subtensor = bt.AsyncSubtensor(config=self.config)
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
        if self.wallet.hotkey.ss58_address not in self.metagraph.hotkeys:
            bt.logging.error(
                f"Your validator: {self.wallet} is not registered to chain connection: {self.subtensor}. Run btcli register --netuid 18 and try again."
            )
            exit()

        await initialize_redis()

    async def sync_available_uids(self):
        while True:
            start_time = time.time()
            try:
                self.available_uids = await self.get_available_uids_is_alive()

                if not hasattr(self, "uid_manager"):
                    self.uid_manager = UIDManager(
                        wallet=self.wallet,
                        metagraph=self.metagraph,
                    )

                self.uid_manager.resync(self.available_uids)

                bt.logging.info(
                    f"Number of available UIDs for periodic update: Amount: {len(self.available_uids)}, UIDs: {self.available_uids}"
                )
            except Exception as e:
                bt.logging.error(
                    f"sync_available_uids Failed to update available UIDs: {e}"
                )
                # Consider whether to continue or break the loop upon certain errors.

            end_time = time.time()
            execution_time = end_time - start_time
            bt.logging.info(
                f"sync_available_uids Execution time for getting available UIDs amount is: {execution_time} seconds"
            )

            await asyncio.sleep(self.config.neuron.update_available_uids_interval)

    async def check_uid(self, axon, uid):
        """Asynchronously check if a UID is available."""
        try:
            dendrite = next(self.dendrites)
            response = await dendrite(axon, IsAlive(), deserialize=False, timeout=10)
            if response.is_success:
                bt.logging.debug(f"UID {uid} is active")
                return axon  # Return the axon info instead of the UID
            else:
                raise Exception(f"UID {uid} is not active")
        except Exception as e:
            bt.logging.debug(f"Checking UID {uid}: {e}\n{traceback.format_exc()}")
            raise e

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

        # Filter out the exceptions and keep the successful results
        available_uids = [
            uid
            for uid, result in zip(tasks.keys(), results)
            if not isinstance(result, Exception)
        ]

        return available_uids

    async def get_uids(
        self,
        strategy=QUERY_MINERS.RANDOM,
        is_only_allowed_miner=False,
        specified_uids=None,
    ):
        if len(self.available_uids) == 0:
            bt.logging.info("No available UIDs, attempting to refresh list.")
            return self.available_uids

        if strategy == QUERY_MINERS.RANDOM:
            uid = self.uid_manager.get_miner_uid()
            bt.logging.info(f"Run uids ---------- Amount: 1 | {uid}")
            return uid
        elif strategy == QUERY_MINERS.ALL:
            # Filter uid_list based on specified_uids and only_allowed_miners
            uid_list = [
                uid
                for uid in self.metagraph.uids
                if (not specified_uids or uid in specified_uids)
                and (
                    not is_only_allowed_miner
                    or self.metagraph.axons[uid].coldkey
                    in self.config.neuron.only_allowed_miners
                )
            ]

            uids = torch.tensor(uid_list) if uid_list else torch.tensor([])
            bt.logging.info(f"Run uids ---------- Amount: {len(uids)} | {uids}")
            return uids.to(self.config.neuron.device)

    async def update_scores(
        self,
        wandb_data,
        responses,
        uids,
        rewards,
        all_rewards,
        all_original_rewards,
        val_score_responses_list,
        neuron,
        query_type,
    ):
        try:
            if self.config.wandb_on:
                wandb.log(wandb_data)

            weights = await get_weights(self)

            asyncio.create_task(
                save_logs_in_chunks(
                    self,
                    responses=responses,
                    uids=uids,
                    rewards=rewards,
                    twitter_rewards=all_rewards[0],
                    search_rewards=all_rewards[1],
                    summary_rewards=all_rewards[2],
                    performance_rewards=all_rewards[3],
                    original_twitter_rewards=all_original_rewards[0],
                    original_search_rewards=all_original_rewards[1],
                    original_summary_rewards=all_original_rewards[2],
                    original_performance_rewards=all_original_rewards[3],
                    tweet_scores=val_score_responses_list[0],
                    search_scores=val_score_responses_list[1],
                    summary_link_scores=val_score_responses_list[2],
                    weights=weights,
                    neuron=neuron,
                    netuid=self.config.netuid,
                    query_type=query_type,
                )
            )
        except Exception as e:
            bt.logging.error(f"Error in update_scores: {e}")
            raise e

    async def update_scores_for_basic(
        self,
        wandb_data,
        responses,
        uids,
        rewards,
        all_rewards,
        all_original_rewards,
        val_score_responses_list,
        neuron,
    ):
        try:
            if self.config.wandb_on:
                wandb.log(wandb_data)

            # weights = (
            #     await self.run_sync_in_async(lambda: get_weights(self))
            # )

            # asyncio.create_task(
            #     save_logs_in_chunks_for_basic(
            #         self,
            #         responses=responses,
            #         uids=uids,
            #         rewards=rewards,
            #         twitter_rewards=all_rewards[0],
            #         performance_rewards=all_rewards[1],
            #         original_twitter_rewards=all_original_rewards[0],
            #         original_performance_rewards=all_original_rewards[1],
            #         tweet_scores=val_score_responses_list[0],
            #         weights=weights,
            #         neuron=neuron,
            #         netuid=self.config.netuid,
            #     )
            # )
        except Exception as e:
            bt.logging.error(f"Error in update_scores_for_basic: {e}")
            raise e

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

            alpha = self.config.neuron.moving_average_alpha
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

    async def compute_organic_responses(self, validator):
        specified_uids = await validator.get_uids_with_no_history(
            self.metagraph.uids.tolist()
        )

        if specified_uids:
            bt.logging.info(
                f"Running {validator.__class__.__name__} synthetic queries with specified uids: {specified_uids}"
            )

            # Call the appropriate query function based on validator type
            await validator.query_and_score(
                strategy=QUERY_MINERS.ALL, specified_uids=specified_uids
            )

        random_organic_responses = await validator.get_random_organic_responses()

        # Compute rewards and penalties using random organic responses
        await validator.compute_rewards_and_penalties(
            **random_organic_responses,
            start_time=time.time(),
            is_synthetic=True,
        )

    async def blocks_until_next_epoch(self):
        try:
            current_block = await self.subtensor.get_current_block()
        except Exception as e:
            bt.logging.error(
                f"Error getting current block: {e}, reinitializing subtensor..."
            )

            await self.subtensor.close()

            self.subtensor = bt.AsyncSubtensor(config=self.config)
            self.metagraph = await self.subtensor.metagraph(self.config.netuid)
            current_block = await self.subtensor.get_current_block()

        tempo = await self.subtensor.tempo(self.config.netuid, current_block)

        return tempo - (current_block + self.config.netuid + 1) % (tempo + 1)

    async def sync_metagraph(self):
        while True:
            try:
                await asyncio.sleep(30 * 60)  # 30 minutes

                sync_start_time = time.time()

                bt.logging.info("Calling sync metagraph method")
                await resync_metagraph(self)
                bt.logging.info("Completed calling sync metagraph method")

                sync_end_time = time.time()
                bt.logging.info(
                    f"Sync metagraph method execution time: {sync_end_time - sync_start_time:.2f} seconds"
                )

                # Ensure validator hotkey is still registered on the network.
                await self.check_registered()
            except Exception as e:
                bt.logging.error(f"Error in sync_metagraph: {e}")

    async def sync(self):
        """
        Wrapper for synchronizing the state of the network for the given miner or validator.
        """

        while True:
            try:
                blocks_left = await self.blocks_until_next_epoch()

                bt.logging.debug(f"Blocks left until next epoch: {blocks_left}")

                if blocks_left <= 20 and self.should_set_weights():
                    weight_set_start_time = time.time()
                    bt.logging.info("Setting weights as per condition.")
                    await set_weights(self)
                    weight_set_end_time = time.time()
                    bt.logging.info(
                        f"Weight setting execution time: {weight_set_end_time - weight_set_start_time:.2f} seconds"
                    )
                    await asyncio.sleep(300)

                if self.config.neuron.synthetic_disabled:
                    if blocks_left <= 100:
                        if not self.organic_responses_computed:
                            bt.logging.info("Computing organic responses")

                            random_validator = random.choices(
                                [
                                    self.advanced_scraper_validator,
                                    self.x_scraper_validator,
                                ],
                                weights=[0.6, 0.4],
                            )[0]

                            self.loop.create_task(
                                self.compute_organic_responses(random_validator)
                            )

                            self.organic_responses_computed = True
                        else:
                            bt.logging.info(
                                "Skipping compute organic responses: Already executed."
                            )
                    else:
                        self.organic_responses_computed = False

            except Exception as e:
                bt.logging.error(f"Error in validator sync: {e}")

            await asyncio.sleep(60)

    async def check_registered(self):
        # --- Check for registration
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
        bt.logging.info("Starting Neuron")

        await self.initialize_components()
        await self.check_registered()

        self.loop = asyncio.get_event_loop()

        init_wandb(self)

        # Init Weights.
        bt.logging.debug("loading", "moving_averaged_scores")
        self.moving_averaged_scores = await load_moving_averaged_scores(
            self.metagraph, self.config
        )
        bt.logging.debug(str(self.moving_averaged_scores))

        self.loop.create_task(self.sync_metagraph())
        self.loop.create_task(self.sync())
        self.loop.create_task(self.sync_available_uids())

        try:
            self.start_query_tasks()
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

        await self.subtensor.close()

        for dendrite in self.dendrite_list:
            dendrite.close_session()
