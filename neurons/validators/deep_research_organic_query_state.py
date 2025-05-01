from typing import List
import random

from datura.protocol import DeepResearchSynapse
from datura.dataset.date_filters import DateFilter, DateFilterType
from neurons.validators.base_organic_query_state import BaseOrganicQueryState


class DeepResearchOrganicQueryState(BaseOrganicQueryState):
    def save_organic_queries(
        self,
        final_synapses: List[DeepResearchSynapse],
        uids,
        original_rewards,
    ):
        """Save the deep research organic queries and their rewards for future reference"""

        content_rewards = original_rewards[0]
        data_rewards = original_rewards[1]
        logical_coherence_rewards = original_rewards[2]
        source_links_rewards = original_rewards[3]
        system_message_rewards = original_rewards[4]
        performance_rewards = original_rewards[5]

        for (
            uid_tensor,
            synapse,
            content_reward,
            data_reward,
            logical_coherence_reward,
            source_links_reward,
            system_message_reward,
            performance_reward,
        ) in zip(
            uids,
            final_synapses,
            content_rewards,
            data_rewards,
            logical_coherence_rewards,
            source_links_rewards,
            system_message_rewards,
            performance_rewards,
        ):
            uid = uid_tensor.item()
            hotkey = synapse.axon.hotkey

            is_failed_organic = False

            # Check if organic query failed by rewards
            if 0 in [
                content_reward,
                data_reward,
                logical_coherence_reward,
                source_links_reward,
                system_message_reward,
                performance_reward,
            ]:
                is_failed_organic = True

            # Save penalty for the miner for the next synthetic query
            if is_failed_organic:
                self.record_failed_organic_query(uid, hotkey)

            self.save_organic_query_history(hotkey, synapse, is_failed_organic)

    def get_random_organic_query(self, uids, neurons):
        """Gets a random organic query from the history to score with other miners"""
        failed_synapses = self.collect_failed_synapses()

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

        self.clear_history()

        return synapse, query, synapse_uid, specified_uids
