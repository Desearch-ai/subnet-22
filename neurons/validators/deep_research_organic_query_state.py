from typing import List
import random
from datura.protocol import DeepResearchSynapse
from datura.dataset.date_filters import DateFilter, DateFilterType
from datetime import datetime
import pytz
import bittensor as bt


class DeepResearchOrganicQueryState:
    def __init__(self) -> None:
        # Tracks failed organic queries and in the next synthetic query, we will penalize the miner
        self.organic_penalties = {}

        # Tracks the all organic synapses
        self.organic_history = {}

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

            # axon = next(axon for axon in axons if axon.hotkey == synapse.axon.hotkey)

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
                bt.logging.info(
                    f"Failed deep research organic query by miner UID: {uid}, Hotkey: {hotkey}"
                )

                self.organic_penalties[hotkey] = (
                    self.organic_penalties.get(hotkey, 0) + 1
                )

            if not hotkey in self.organic_history:
                self.organic_history[hotkey] = []

            self.organic_history[hotkey].append((synapse, is_failed_organic))

    def has_penalty(self, hotkey: str) -> bool:
        """Check if the miner has a penalty and decrement it"""
        penalties = self.organic_penalties.get(hotkey, 0)

        if penalties > 0:
            self.organic_penalties[hotkey] -= 1
            return True

        return False

    def get_random_organic_query(self, uids, neurons):
        """Gets a random organic query from the history to score with other miners"""
        # Collect all failed synapses
        failed_synapses = []

        for hotkey, synapses in self.organic_history.items():
            failed_synapses.extend(
                [(hotkey, synapse) for synapse, is_failed in synapses if is_failed]
            )

        # If there are no failed synapses, collect all synapses
        if not failed_synapses:
            for hotkey, synapses in self.organic_history.items():
                failed_synapses.extend([(hotkey, synapse) for synapse, _ in synapses])

        # If there are still no synapses, return None
        if not failed_synapses:
            return None

        # Choose a random synapse
        hotkey, synapse = random.choice(failed_synapses)

        start_date = datetime.strptime(
            synapse.start_date, "%Y-%m-%dT%H:%M:%SZ"
        ).replace(tzinfo=pytz.utc)

        end_date = datetime.strptime(synapse.end_date, "%Y-%m-%dT%H:%M:%SZ").replace(
            tzinfo=pytz.utc
        )

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

        # All miners to call except the one that me the query
        specified_uids = [uid for uid in uids if uid != synapse_uid]

        self.organic_history = {}

        return synapse, query, synapse_uid, specified_uids

    def remove_deregistered_hotkeys(self, axons):
        """Called after metagraph resync to remove any hotkeys that are no longer registered"""
        hotkeys = [axon.hotkey for axon in axons]

        original_history_count = len(self.organic_history)
        original_penalties_count = len(self.organic_penalties)

        self.organic_history = {
            hotkey: synapses
            for hotkey, synapses in self.organic_history.items()
            if hotkey in hotkeys
        }

        self.organic_penalties = {
            hotkey: penalty
            for hotkey, penalty in self.organic_penalties.items()
            if hotkey in hotkeys
        }

        log_data = {
            "organic_history": original_history_count - len(self.organic_history),
            "organic_penalties": original_penalties_count - len(self.organic_penalties),
        }

        bt.logging.info(
            f"Removed deregistered hotkeys from deep research organic query state: {log_data}"
        )
