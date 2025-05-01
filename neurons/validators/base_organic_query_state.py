from typing import List, Tuple, Dict, Any, Optional, Type
import random
import bittensor as bt
from datetime import datetime
import pytz


class BaseOrganicQueryState:
    """
    Base class for organic query state tracking.
    Provides common functionality for tracking penalties and query history.
    """

    def __init__(self) -> None:
        # Tracks failed organic queries and in the next synthetic query, we will penalize the miner
        self.organic_penalties: Dict[str, int] = {}

        # Tracks all organic synapses
        self.organic_history: Dict[str, List[Tuple[Any, bool]]] = {}

    def has_penalty(self, hotkey: str) -> bool:
        """Check if the miner has a penalty and decrement it"""
        penalties = self.organic_penalties.get(hotkey, 0)

        if penalties > 0:
            self.organic_penalties[hotkey] -= 1
            return True

        return False

    def record_failed_organic_query(self, uid: int, hotkey: str) -> None:
        """Record a failed organic query and increment penalty counter"""
        bt.logging.info(f"Failed organic query by miner UID: {uid}, Hotkey: {hotkey}")
        self.organic_penalties[hotkey] = self.organic_penalties.get(hotkey, 0) + 1

    def remove_deregistered_hotkeys(self, axons) -> None:
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
            f"Removed deregistered hotkeys from organic query state: {log_data}"
        )

    def save_organic_query_history(
        self, hotkey: str, synapse: Any, is_failed: bool
    ) -> None:
        """Save a synapse and its failed state to the history"""
        if hotkey not in self.organic_history:
            self.organic_history[hotkey] = []
        self.organic_history[hotkey].append((synapse, is_failed))

    def collect_failed_synapses(self) -> List[Tuple[str, Any]]:
        """Collect all failed synapses from history"""
        failed_synapses = []

        for hotkey, synapses in self.organic_history.items():
            failed_synapses.extend(
                [(hotkey, synapse) for synapse, is_failed in synapses if is_failed]
            )

        # If there are no failed synapses, collect all synapses
        if not failed_synapses:
            for hotkey, synapses in self.organic_history.items():
                failed_synapses.extend([(hotkey, synapse) for synapse, _ in synapses])

        return failed_synapses

    def parse_datetime(self, date_str: str, format_str: str) -> datetime:
        """Parse a datetime string into a datetime object with UTC timezone"""
        return datetime.strptime(date_str, format_str).replace(tzinfo=pytz.utc)

    def get_specified_uids(self, uids, synapse_uid) -> List[int]:
        """Get all uids except the one that made the query"""
        return [uid for uid in uids if uid != synapse_uid]

    def clear_history(self) -> None:
        """Clear the organic history"""
        self.organic_history = {}
