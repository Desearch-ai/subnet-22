import asyncio
import random
import time
import traceback

import bittensor as bt

from desearch import QUERY_MINERS


class SyntheticQueryRunnerMixin:
    """
    Base class that handles running synthetic and organic queries
    at specified intervals.
    """

    async def run_synthetic_queries(self, validator, strategy):
        """
        Run synthetic queries using the provided validator and strategy.

        Args:
            validator: The validator to use for querying
            strategy: The query strategy (RANDOM or ALL)
        """
        bt.logging.info(
            f"Starting run_synthetic_queries with validator={validator}, strategy={strategy}"
        )
        total_start_time = time.time()

        try:
            start_time = time.time()

            bt.logging.info(
                f"Running step forward for query_synapse, Step: {self.step}"
            )

            await self.run_query_and_score(validator, strategy)

            end_time = time.time()

            bt.logging.info(
                f"Completed gathering coroutines for run_query_and_score in {end_time - start_time:.2f} seconds"
            )

            self.step += 1
            bt.logging.info(f"Incremented step to {self.step}")
        except Exception as err:
            bt.logging.error("Error in run_synthetic_queries", str(err))
            bt.logging.debug(
                traceback.format_exception(type(err), err, err.__traceback__)
            )
        finally:
            total_end_time = time.time()
            total_execution_time = (total_end_time - total_start_time) / 60
            bt.logging.info(
                f"Total execution time for run_synthetic_queries: {total_execution_time:.2f} minutes"
            )

    async def run_query_and_score(self, validator, strategy):
        """
        Query a synapse using the provided validator and strategy.

        Args:
            validator: The validator to use for querying
            strategy: The query strategy to use
        """
        try:
            await validator.query_and_score(strategy)
        except Exception as e:
            bt.logging.error(f"General exception: {e}\n{traceback.format_exc()}")
            await asyncio.sleep(100)

    async def run_with_interval(self, interval, strategy):
        """
        Run synthetic queries at specified intervals with the given strategy.

        Args:
            interval: Time in seconds between query runs
            strategy: The query strategy (RANDOM or ALL)
        """
        while True:
            try:
                if not self.available_uids:
                    bt.logging.info("No available UIDs, sleeping for 5 seconds.")
                    await asyncio.sleep(5)
                    continue

                choice = random.choices(
                    [
                        self.advanced_scraper_validator,
                        self.x_scraper_validator,
                    ],
                    weights=[0.6, 0.4],
                )[0]

                self.loop.create_task(self.run_synthetic_queries(choice, strategy))

                await asyncio.sleep(interval)  # Wait for synthetic interval
            except Exception as e:
                bt.logging.error(f"Error during task execution: {e}")
                await asyncio.sleep(interval)  # Wait before retrying

    def start_query_tasks(self):
        """
        Start all query tasks based on configuration.
        """

        if not self.config.neuron.synthetic_disabled:
            if self.config.neuron.run_random_miner_syn_qs_interval > 0:
                self.loop.create_task(
                    self.run_with_interval(
                        self.config.neuron.run_all_miner_syn_qs_interval,
                        QUERY_MINERS.RANDOM,
                    )
                )

            if self.config.neuron.run_all_miner_syn_qs_interval > 0:
                self.loop.create_task(
                    self.run_with_interval(
                        self.config.neuron.run_all_miner_syn_qs_interval,
                        QUERY_MINERS.ALL,
                    )
                )
