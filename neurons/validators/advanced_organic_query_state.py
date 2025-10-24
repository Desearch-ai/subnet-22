from typing import List
import random

from desearch.protocol import ScraperStreamingSynapse
from desearch.dataset.date_filters import DateFilter, DateFilterType
from neurons.validators.base_organic_query_state import BaseOrganicQueryState


class AdvancedOrganicQueryState(BaseOrganicQueryState):
    async def save_organic_queries(
        self,
        final_synapses: List[ScraperStreamingSynapse],
        uids,
        original_rewards,
    ):
        """Save the organic queries and their rewards for future reference"""

        twitter_rewards = original_rewards[0]
        search_rewards = original_rewards[1]
        summary_rewards = original_rewards[2]
        performance_rewards = original_rewards[3]

        for (
            uid_tensor,
            synapse,
            twitter_reward,
            search_reward,
            summary_reward,
            performance_reward,
        ) in zip(
            uids,
            final_synapses,
            twitter_rewards,
            search_rewards,
            summary_rewards,
            performance_rewards,
        ):
            uid = uid_tensor.item()
            hotkey = synapse.axon.hotkey

            is_twitter_search = "Twitter Search" in synapse.tools
            is_web_search = any(
                tool in synapse.tools
                for tool in [
                    "Web Search",
                    "Wikipedia Search",
                    "Youtube Search",
                    "ArXiv Search",
                    "Reddit Search",
                    "Hacker News Search",
                ]
            )

            is_failed_organic = False

            # Check if organic query failed by rewards
            if (
                (performance_reward == 0 or summary_reward == 0)
                or (is_twitter_search and twitter_reward == 0)
                or (is_web_search and search_reward == 0)
            ):
                is_failed_organic = True

            # Save penalty for the miner for the next synthetic query
            if is_failed_organic:
                await self.record_failed_organic_query(uid, hotkey)

            await self.save_organic_query_history(hotkey, synapse, is_failed_organic)

    async def get_random_organic_query(self, uids, neurons):
        """Gets a random organic query from the history to score with other miners"""
        failed_synapses = await self.collect_failed_synapses()

        # If there are still no synapses, return None
        if not failed_synapses:
            return None

        # Choose a random synapse
        hotkey, synapse = random.choice(failed_synapses)

        start_date = self.parse_datetime(synapse.start_date, "%Y-%m-%dT%H:%M:%SZ")
        end_date = self.parse_datetime(synapse.end_date, "%Y-%m-%dT%H:%M:%SZ")

        date_filter = DateFilter(
            start_date=start_date,
            end_date=end_date,
            date_filter_type=DateFilterType(synapse.date_filter_type),
        )

        neuron = next(neuron for neuron in neurons if neuron.hotkey == hotkey)
        synapse_uid = neuron.uid

        query = {
            "content": synapse.prompt,
            "tools": synapse.tools,
            "date_filter": date_filter,
        }

        # All miners to call except the one that made the query
        specified_uids = self.get_specified_uids(uids, synapse_uid)

        await self.clear_history()

        return synapse, query, synapse_uid, specified_uids
