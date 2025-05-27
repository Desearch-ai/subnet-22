import asyncio
import random
import bittensor as bt
from datura import QUERY_MINERS
import traceback
import time


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

    async def run_organic_queries(self, validator):
        """Run organic queries using the advanced scraper validator."""
        result = await validator.organic_query_state.get_random_organic_query(
            self.available_uids, self.metagraph.neurons
        )

        if not result:
            bt.logging.info("No organic queries are in history to run")
            return

        synapse, query, synapse_uid, specified_uids = result

        bt.logging.info(f"Running organic queries for synapse: {synapse}")

        async for _ in validator.organic(
            query=query,
            model=synapse.model if hasattr(synapse, "model") else None,
            random_synapse=synapse,
            random_uid=synapse_uid,
            specified_uids=specified_uids,
        ):
            pass

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
                        self.basic_scraper_validator,
                        self.deep_research_validator,
                        self.people_search_validator,
                    ],
                    weights=[0.5, 0.20, 0.15, 0.15],
                )[0]

                self.loop.create_task(self.run_synthetic_queries(choice, strategy))

                await asyncio.sleep(interval)  # Wait for synthetic interval
            except Exception as e:
                bt.logging.error(f"Error during task execution: {e}")
                await asyncio.sleep(interval)  # Wait before retrying

    async def run_organic_with_interval(self, interval):
        """
        Run organic queries at specified intervals.

        Args:
            interval: Time in seconds between query runs
        """
        while True:
            try:
                if not self.available_uids:
                    await asyncio.sleep(5)
                    continue

                self.loop.create_task(
                    self.run_organic_queries(self.advanced_scraper_validator)
                )

                self.loop.create_task(
                    self.run_organic_queries(self.basic_scraper_validator)
                )

                await asyncio.sleep(interval)
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

            # Run organic queries every three hours
            three_hours_in_seconds = 10800
            self.loop.create_task(
                self.run_organic_with_interval(three_hours_in_seconds)
            )
