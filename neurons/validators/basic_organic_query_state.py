from typing import List
import random

import bittensor as bt

from datura.protocol import (
    TwitterSearchSynapse,
    TwitterIDSearchSynapse,
    TwitterURLsSearchSynapse,
    WebSearchSynapse,
)
from neurons.validators.base_organic_query_state import BaseOrganicQueryState


class BasicOrganicQueryState(BaseOrganicQueryState):
    async def save_organic_queries(
        self,
        final_synapses: List[bt.Synapse],
        uids,
        original_rewards,
    ):
        """Save the organic queries and their rewards for future reference"""

        twitter_rewards = original_rewards[0]
        performance_rewards = original_rewards[1]

        for (
            uid_tensor,
            synapse,
            twitter_reward,
            performance_reward,
        ) in zip(
            uids,
            final_synapses,
            twitter_rewards,
            performance_rewards,
        ):
            uid = uid_tensor.item()
            hotkey = synapse.axon.hotkey

            # Instead of checking synapse.tools, we now check the class
            is_twitter_search = isinstance(
                synapse,
                (
                    TwitterSearchSynapse,
                    TwitterIDSearchSynapse,
                    TwitterURLsSearchSynapse,
                ),
            )
            is_web_search = isinstance(synapse, WebSearchSynapse)

            is_failed_organic = False

            # Check if organic query failed by rewards
            if (
                performance_reward == 0
                or (is_twitter_search and twitter_reward == 0)
                or (is_web_search)
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

        # Randomly pick one
        hotkey, synapse = random.choice(failed_synapses)

        # Check synapse class type and gather content
        content = None
        start_date = None
        end_date = None

        if isinstance(synapse, TwitterSearchSynapse):
            # The 'query' field can be used as content
            content = synapse.query

            # Convert string (YYYY-MM-DD) to datetime objects, if present
            if synapse.start_date:
                start_date = self.parse_datetime(synapse.start_date, "%Y-%m-%d")

            if synapse.end_date:
                end_date = self.parse_datetime(synapse.end_date, "%Y-%m-%d")

        elif isinstance(synapse, TwitterIDSearchSynapse):
            content = synapse.id

        elif isinstance(synapse, TwitterURLsSearchSynapse):
            # 'urls' is a dict
            content = synapse.urls

        elif isinstance(synapse, WebSearchSynapse):
            content = synapse.query

        # Find the neuron's UID
        neuron = next(n for n in neurons if n.hotkey == hotkey)
        synapse_uid = neuron.uid

        # Build the final query
        query = {"query": content}
        if isinstance(synapse, TwitterSearchSynapse):
            query["start_date"] = start_date
            query["end_date"] = end_date

        # Identify other miners except the one that made the query
        specified_uids = self.get_specified_uids(uids, synapse_uid)

        # Clear all history
        await self.clear_history()

        return synapse, query, synapse_uid, specified_uids
