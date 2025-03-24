import random
import torch
import wandb
import asyncio
import concurrent
import traceback
import copy
import bittensor as bt
import time
import sys
from datura.protocol import IsAlive
from datura.bittensor.dendrite import Dendrite
from datura.bittensor.subtensor import Subtensor
from datura.bittensor.wallet import Wallet
from neurons.validators.advanced_scraper_validator import AdvancedScraperValidator
from neurons.validators.basic_scraper_validator import BasicScraperValidator
from neurons.validators.basic_web_scraper_validator import BasicWebScraperValidator
from neurons.validators.config import add_args, check_config, config
from neurons.validators.weights import init_wandb, set_weights, get_weights
from traceback import print_exception
from neurons.validators.base_validator import AbstractNeuron
from datura import QUERY_MINERS
from datura.misc import ttl_get_block
from datura.utils import (
    resync_metagraph,
    save_logs_in_chunks,
)
from datura.redis.utils import load_moving_averaged_scores, save_moving_averaged_scores
from neurons.validators.proxy.uid_manager import UIDManager


class Neuron(AbstractNeuron):
    @classmethod
    def check_config(cls, config: "bt.Config"):
        check_config(cls, config)

    @classmethod
    def add_args(cls, parser):
        add_args(cls, parser)

    @classmethod
    def config(cls):
        return config(cls)

    subtensor: "bt.subtensor"
    wallet: "bt.wallet"
    metagraph: "bt.metagraph"
    dendrite: "bt.dendrite"

    loop: asyncio.AbstractEventLoop

    advanced_scraper_validator: "AdvancedScraperValidator"
    basic_scraper_validator: "BasicScraperValidator"
    basic_web_scraper_validator: "BasicWebScraperValidator"
    moving_average_scores: torch.Tensor = None
    uid: int = None
    shutdown_event: asyncio.Event()

    uid_manager: UIDManager

    @property
    def block(self):
        return ttl_get_block(self)

    def __init__(self):
        self.config = Neuron.config()
        self.check_config(self.config)
        bt.logging(config=self.config, logging_dir=self.config.neuron.full_path)
        print(self.config)
        bt.logging.info("neuron.__init__()")

        self.initialize_components()

        init_wandb(self)

        self.advanced_scraper_validator = AdvancedScraperValidator(neuron=self)
        self.basic_scraper_validator = BasicScraperValidator(neuron=self)
        self.basic_web_scraper_validator = BasicWebScraperValidator(neuron=self)
        bt.logging.info("initialized_validators")

        self.step = 0
        self.check_registered()

        self.organic_responses_computed = False

        # Init Weights.
        bt.logging.debug("loading", "moving_averaged_scores")
        self.moving_averaged_scores = load_moving_averaged_scores(
            self.metagraph, self.config
        )
        bt.logging.debug(str(self.moving_averaged_scores))
        self.available_uids = []
        self.thread_executor = concurrent.futures.ThreadPoolExecutor(
            thread_name_prefix="asyncio"
        )

    async def run_sync_in_async(self, fn):
        return await self.loop.run_in_executor(self.thread_executor, fn)

    def initialize_components(self):
        bt.logging(config=self.config, logging_dir=self.config.full_path)
        bt.logging.info(
            f"Running validator for subnet: {self.config.netuid} on network: {self.config.subtensor.chain_endpoint}"
        )
        if self.config.neuron.offline:
            self.wallet = Wallet(config=self.config)
            self.subtensor = Subtensor(config=self.config)
            self.metagraph = self.subtensor.metagraph(self.config.netuid)
            self.hotkeys = copy.deepcopy(self.metagraph.hotkeys)
            self.dendrite = Dendrite(wallet=self.wallet)
            self.dendrite1 = Dendrite(wallet=self.wallet)
            self.dendrite2 = Dendrite(wallet=self.wallet)
            self.dendrite3 = Dendrite(wallet=self.wallet)
        else:
            self.wallet = bt.wallet(config=self.config)
            self.subtensor = bt.subtensor(config=self.config)
            self.metagraph = self.subtensor.metagraph(self.config.netuid)
            self.hotkeys = copy.deepcopy(self.metagraph.hotkeys)
            self.dendrite = bt.dendrite(wallet=self.wallet)
            self.dendrite1 = bt.dendrite(wallet=self.wallet)
            self.dendrite2 = bt.dendrite(wallet=self.wallet)
            self.dendrite3 = bt.dendrite(wallet=self.wallet)

        self.uid = self.metagraph.hotkeys.index(self.wallet.hotkey.ss58_address)
        if self.wallet.hotkey.ss58_address not in self.metagraph.hotkeys:
            bt.logging.error(
                f"Your validator: {self.wallet} is not registered to chain connection: {self.subtensor}. Run btcli register --netuid 18 and try again."
            )
            exit()

    async def update_available_uids_periodically(self):
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
                self.advanced_scraper_validator.organic_query_state.remove_deregistered_hotkeys(
                    self.metagraph.axons
                )
                self.basic_scraper_validator.basic_organic_query_state.remove_deregistered_hotkeys(
                    self.metagraph.axons
                )

                bt.logging.info(
                    f"Number of available UIDs for periodic update: Amount: {len(self.available_uids)}, UIDs: {self.available_uids}"
                )
            except Exception as e:
                bt.logging.error(
                    f"update_available_uids_periodically Failed to update available UIDs: {e}"
                )
                # Consider whether to continue or break the loop upon certain errors.

            end_time = time.time()
            execution_time = end_time - start_time
            bt.logging.info(
                f"update_available_uids_periodically Execution time for getting available UIDs amount is: {execution_time} seconds"
            )

            await asyncio.sleep(self.config.neuron.update_available_uids_interval)

    async def check_uid(self, axon, uid):
        """Asynchronously check if a UID is available."""
        try:
            response = await self.dendrite(
                axon, IsAlive(), deserialize=False, timeout=15
            )
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

        # Filter uid_list based on specified_uids and only_allowed_miners
        uid_list = [
            uid
            for uid in self.available_uids
            if (not specified_uids or uid in specified_uids)
            and (
                not is_only_allowed_miner
                or self.metagraph.axons[uid].coldkey
                in self.config.neuron.only_allowed_miners
            )
        ]

        if strategy == QUERY_MINERS.RANDOM:
            uid = self.uid_manager.get_miner_uid()
            uids = torch.tensor([uid]) if uid else torch.tensor([])
        elif strategy == QUERY_MINERS.ALL:
            uids = torch.tensor(uid_list) if uid_list else torch.tensor([])
        bt.logging.info(f"Run uids ---------- Amount: {len(uids)} | {uids}")
        # uid_list = list(available_uids.keys())
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
        organic_penalties,
        neuron,
        query_type,
    ):
        try:
            if self.config.wandb_on:
                wandb.log(wandb_data)

            weights = await self.run_sync_in_async(lambda: get_weights(self))

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
                    organic_penalties=organic_penalties,
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
        organic_penalties,
        neuron,
    ):
        try:
            if self.config.wandb_on:
                wandb.log(wandb_data)

            # weights = await self.run_sync_in_async(lambda: get_weights(self))

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
            #         organic_penalties=organic_penalties,
            #     )
            # )
        except Exception as e:
            bt.logging.error(f"Error in update_scores_for_basic: {e}")
            raise e

    def update_moving_averaged_scores(self, uids, rewards):
        try:
            # Ensure uids is a tensor
            if not isinstance(uids, torch.Tensor):
                uids = torch.tensor(
                    uids, dtype=torch.long, device=self.config.neuron.device
                )

            # Ensure rewards is also a tensor and on the correct device
            if not isinstance(rewards, torch.Tensor):
                rewards = torch.tensor(rewards, device=self.config.neuron.device)

            scattered_rewards = self.moving_averaged_scores.scatter(
                0, uids, rewards
            ).to(self.config.neuron.device)
            average_reward = torch.mean(scattered_rewards)
            bt.logging.info(
                f"Scattered reward: {average_reward:.6f}"
            )  # Rounds to 6 decimal places for logging

            alpha = self.config.neuron.moving_average_alpha
            self.moving_averaged_scores = alpha * scattered_rewards + (
                1 - alpha
            ) * self.moving_averaged_scores.to(self.config.neuron.device)
            save_moving_averaged_scores(self.moving_averaged_scores)
            bt.logging.info(
                f"Moving averaged scores: {torch.mean(self.moving_averaged_scores):.6f}"
            )  # Rounds to 6 decimal places for logging
            return scattered_rewards
        except Exception as e:
            bt.logging.error(f"Error in update_moving_averaged_scores: {e}")
            raise e

    async def query_synapse(self, strategy):
        try:
            await self.advanced_scraper_validator.query_and_score(strategy)
        except Exception as e:
            bt.logging.error(f"General exception: {e}\n{traceback.format_exc()}")
            await asyncio.sleep(100)

    async def basic_query_synapse(self, strategy):
        try:
            await self.basic_scraper_validator.query_and_score_twitter_basic(strategy)
        except Exception as e:
            bt.logging.error(f"General exception: {e}\n{traceback.format_exc()}")
            await asyncio.sleep(100)

    async def run_synthetic_queries(self, strategy):
        bt.logging.info(f"Starting run_synthetic_queries with strategy={strategy}")
        total_start_time = time.time()

        try:
            start_time = time.time()

            bt.logging.info(
                f"Running step forward for query_synapse, Step: {self.step}"
            )

            await asyncio.gather(*[self.query_synapse(strategy) for _ in range(1)])

            end_time = time.time()

            bt.logging.info(
                f"Completed gathering coroutines for query_synapse in {end_time - start_time:.2f} seconds"
            )

            self.step += 1
            bt.logging.info(f"Incremented step to {self.step}")
        except Exception as err:
            bt.logging.error("Error in run_synthetic_queries", str(err))
            bt.logging.debug(print_exception(type(err), err, err.__traceback__))
        finally:
            total_end_time = time.time()
            total_execution_time = (total_end_time - total_start_time) / 60
            bt.logging.info(
                f"Total execution time for run_synthetic_queries: {total_execution_time:.2f} minutes"
            )

    async def run_basic_synthetic_queries(self, strategy):
        bt.logging.info(
            f"Starting run_basic_synthetic_queries with strategy={strategy}"
        )
        total_start_time = time.time()

        try:
            start_time = time.time()

            bt.logging.info(
                f"Running step forward for basic_query_synapse, Step: {self.step}"
            )

            await asyncio.gather(
                *[self.basic_query_synapse(strategy) for _ in range(1)]
            )

            end_time = time.time()

            bt.logging.info(
                f"Completed gathering coroutines for basic_query_synapse in {end_time - start_time:.2f} seconds"
            )

            self.step += 1
            bt.logging.info(f"Incremented step to {self.step}")
        except Exception as err:
            bt.logging.error("Error in run_basic_synthetic_queries", str(err))
            bt.logging.debug(print_exception(type(err), err, err.__traceback__))
        finally:
            total_end_time = time.time()
            total_execution_time = (total_end_time - total_start_time) / 60
            bt.logging.info(
                f"Total execution time for run_basic_synthetic_queries: {total_execution_time:.2f} minutes"
            )

    async def run_organic_queries(self):
        result = self.advanced_scraper_validator.organic_query_state.get_random_organic_query(
            self.available_uids, self.metagraph.neurons
        )

        if not result:
            bt.logging.info("No organic queries are in history to run")
            return

        synapse, query, synapse_uid, specified_uids = result

        bt.logging.info(f"Running organic queries for prompt: {synapse.prompt}")

        async for _ in self.advanced_scraper_validator.organic(
            query=query,
            model=synapse.model,
            random_synapse=synapse,
            random_uid=synapse_uid,
            specified_uids=specified_uids,
        ):
            pass

    async def run_basic_organic_queries(self):
        result = self.basic_scraper_validator.basic_organic_query_state.get_random_organic_query(
            self.available_uids, self.metagraph.neurons
        )

        if not result:
            bt.logging.info("No organic queries are in history to run")
            return

        synapse, query, synapse_uid, specified_uids = result

        bt.logging.info(f"Running organic queries for prompt: {synapse.query}")

        async for _ in self.basic_scraper_validator.organic(
            query=query,
            random_synapse=synapse,
            random_uid=synapse_uid,
            specified_uids=specified_uids,
        ):
            pass

    async def compute_organic_responses(self):
        specified_uids = self.advanced_scraper_validator.get_uids_with_no_history(
            self.available_uids
        )
        if specified_uids:
            bt.logging.info(
                f"Running advanced synthetic queries with specified uids: {specified_uids}"
            )
            await self.advanced_scraper_validator.query_and_score(
                strategy=QUERY_MINERS.ALL, specified_uids=specified_uids
            )

        await self.advanced_scraper_validator.compute_rewards_and_penalties(
            **self.advanced_scraper_validator.get_random_organic_responses(),
            start_time=time.time(),
        )

    async def compute_basic_organic_responses(self):
        specified_uids = self.basic_scraper_validator.get_uids_with_no_history(
            self.available_uids
        )

        if specified_uids:
            bt.logging.info(
                f"Running basic synthetic queries with specified uids: {specified_uids}"
            )
            await self.basic_scraper_validator.query_and_score_twitter_basic(
                strategy=QUERY_MINERS.ALL, specified_uids=specified_uids
            )

        await self.basic_scraper_validator.compute_rewards_and_penalties(
            **self.basic_scraper_validator.get_random_organic_responses(),
            start_time=time.time(),
        )

    async def compute_web_basic_organic_responses(self):
        specified_uids = self.basic_web_scraper_validator.get_uids_with_no_history(
            self.available_uids
        )
        if specified_uids:
            bt.logging.info(
                f"Running basic web synthetic queries with specified uids: {specified_uids}"
            )
            await self.basic_web_scraper_validator.query_and_score_web_basic(
                strategy=QUERY_MINERS.ALL, specified_uids=specified_uids
            )

        await self.basic_web_scraper_validator.compute_rewards_and_penalties(
            **self.basic_web_scraper_validator.get_random_organic_responses(),
            start_time=time.time(),
        )

    async def get_current_block(self):
        return await self.run_sync_in_async(self.subtensor.get_current_block)

    async def blocks_until_next_epoch(self):
        try:
            current_block = await self.get_current_block()
        except Exception as e:
            bt.logging.error(
                f"Error getting current block: {e}, reinitializing subtensor..."
            )

            self.subtensor = bt.subtensor(config=self.config)
            current_block = await self.get_current_block()

        tempo = self.subtensor.tempo(self.config.netuid, current_block)

        return tempo - (current_block + self.config.netuid + 1) % (tempo + 1)

    async def sync_metagraph(self):
        while True:
            await asyncio.sleep(30 * 60)  # 30 minutes

            sync_start_time = time.time()

            bt.logging.info("Calling sync metagraph method")
            await self.run_sync_in_async(lambda: resync_metagraph(self))
            bt.logging.info("Completed calling sync metagraph method")

            sync_end_time = time.time()
            bt.logging.info(
                f"Sync metagraph method execution time: {sync_end_time - sync_start_time:.2f} seconds"
            )

            # Ensure validator hotkey is still registered on the network.
            self.check_registered()

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
                    await self.run_sync_in_async(lambda: set_weights(self))
                    weight_set_end_time = time.time()
                    bt.logging.info(
                        f"Weight setting execution time: {weight_set_end_time - weight_set_start_time:.2f} seconds"
                    )
                    await asyncio.sleep(300)

                if self.config.neuron.synthetic_disabled:
                    if blocks_left <= 100:
                        if not self.organic_responses_computed:
                            bt.logging.info("Computing organic responses")
                            tasks = [
                                self.compute_basic_organic_responses,
                                self.compute_organic_responses,
                                self.compute_web_basic_organic_responses,
                            ]
                            self.loop.create_task(random.choice(tasks)())

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

    def check_registered(self):
        # --- Check for registration
        if not self.subtensor.is_hotkey_registered(
            netuid=self.config.netuid,
            hotkey_ss58=self.wallet.hotkey.ss58_address,
        ):
            bt.logging.error(
                f"Wallet: {self.wallet} is not registered on netuid {self.config.netuid}."
                f" Please register the hotkey using `btcli subnets register` before trying again"
            )
            sys.exit()

    def should_set_weights(self) -> bool:
        # Don't set weights on initialization.
        # if self.step == 0:
        #     bt.logging.info("Skipping weight setting on initialization.")
        #     return False

        # Check if enough epoch blocks have elapsed since the last epoch.
        if self.config.neuron.disable_set_weights:
            bt.logging.info("Weight setting is disabled by configuration.")
            return False

        # Define appropriate logic for when set weights.
        # difference = self.block - self.metagraph.last_update[self.uid]
        # print(
        #     f"Current block: {self.block}, Last update for UID {self.uid}: {self.metagraph.last_update[self.uid]}, Difference: {difference}"
        # )
        # should_set = difference > self.config.neuron.checkpoint_block_length
        # bt.logging.info(f"Should set weights: {should_set}")
        # return should_set
        return True  # Update right not based on interval of synthetic data

    async def run(self):
        self.loop = asyncio.get_event_loop()

        self.loop.create_task(self.sync_metagraph())
        self.loop.create_task(self.sync())
        self.loop.create_task(self.update_available_uids_periodically())
        bt.logging.info(f"Validator starting at block: {self.block}")

        try:

            async def run_with_interval(interval, strategy):
                query_count = 0  # Initialize query count
                while True:
                    try:
                        if not self.available_uids:
                            bt.logging.info(
                                "No available UIDs, sleeping for 10 seconds."
                            )
                            await asyncio.sleep(10)
                            continue

                        if random.choices([True, False], weights=[0.7, 0.3])[0]:
                            self.loop.create_task(self.run_synthetic_queries(strategy))
                        else:
                            self.loop.create_task(
                                self.run_basic_synthetic_queries(strategy)
                            )

                        await asyncio.sleep(interval)  # Wait for synthetic interval
                    except Exception as e:
                        bt.logging.error(f"Error during task execution: {e}")
                        await asyncio.sleep(interval)  # Wait before retrying

            async def run_organic_with_interval(interval):
                while True:
                    try:
                        if not self.available_uids:
                            await asyncio.sleep(10)
                            continue
                        self.loop.create_task(self.run_organic_queries())
                        self.loop.create_task(self.run_basic_organic_queries())

                        await asyncio.sleep(interval)
                    except Exception as e:
                        bt.logging.error(f"Error during task execution: {e}")
                        await asyncio.sleep(interval)  # Wait before retrying

            if not self.config.neuron.synthetic_disabled:
                if self.config.neuron.run_random_miner_syn_qs_interval > 0:
                    self.loop.create_task(
                        run_with_interval(
                            self.config.neuron.run_all_miner_syn_qs_interval,
                            QUERY_MINERS.RANDOM,
                        )
                    )

                if self.config.neuron.run_all_miner_syn_qs_interval > 0:
                    self.loop.create_task(
                        run_with_interval(
                            self.config.neuron.run_all_miner_syn_qs_interval,
                            QUERY_MINERS.ALL,
                        )
                    )
            # If someone intentionally stops the validator, it'll safely terminate operations.

            three_hours_in_seconds = 10800
            self.loop.create_task(run_organic_with_interval(three_hours_in_seconds))

        except KeyboardInterrupt:
            self.axon.stop()
            bt.logging.success("Validator killed by keyboard interrupt.")
            sys.exit()

        # In case of unforeseen errors, the validator will log the error and quit
        except Exception as err:
            bt.logging.error("Error during validation", str(err))
            bt.logging.debug(print_exception(type(err), err, err.__traceback__))
            self.should_exit = True


def main():
    asyncio.run(Neuron().run())


if __name__ == "__main__":
    main()
