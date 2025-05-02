from typing import List, Tuple, Any
import bittensor as bt
from datetime import datetime
import pytz
import json
import jsonpickle
from datura.redis.redis_client import redis_client


class BaseOrganicQueryState:
    """
    Base class for organic query state tracking.
    Provides common functionality for tracking penalties and query history using Redis.
    """

    def __init__(self) -> None:
        # Redis keys with class name as prefix for namespace isolation
        self.prefix = self.__class__.__name__
        self.penalties_key = f"{self.prefix}:organic_penalties"
        self.history_key = f"{self.prefix}:organic_history"

    def has_penalty(self, hotkey: str) -> bool:
        """Check if the miner has a penalty and decrement it with atomic operations"""
        penalties = redis_client.hget(self.penalties_key, hotkey)
        penalties = int(penalties) if penalties else 0

        if penalties > 0:
            # Use atomic decrement
            redis_client.hincrby(self.penalties_key, hotkey, -1)
            return True
        return False

    def record_failed_organic_query(self, uid: int, hotkey: str) -> None:
        """Record a failed organic query and increment penalty counter atomically"""
        bt.logging.info(f"Failed organic query by miner UID: {uid}, Hotkey: {hotkey}")
        # Use atomic increment
        redis_client.hincrby(self.penalties_key, hotkey, 1)

    def remove_deregistered_hotkeys(self, axons) -> None:
        """Called after metagraph resync to remove any hotkeys that are no longer registered"""
        hotkeys = [axon.hotkey for axon in axons]

        # Get all current hotkeys in redis
        organic_history_hotkeys = redis_client.hkeys(self.history_key)
        organic_penalties_hotkeys = redis_client.hkeys(self.penalties_key)

        original_history_count = len(organic_history_hotkeys)
        original_penalties_count = len(organic_penalties_hotkeys)

        # Remove hotkeys not in the axon list (pipeline for efficiency)
        pipe = redis_client.pipeline()
        for hotkey in organic_history_hotkeys:
            if hotkey not in hotkeys:
                pipe.hdel(self.history_key, hotkey)

        for hotkey in organic_penalties_hotkeys:
            if hotkey not in hotkeys:
                pipe.hdel(self.penalties_key, hotkey)

        pipe.execute()

        # Count how many were removed
        current_history_count = len(redis_client.hkeys(self.history_key))
        current_penalties_count = len(redis_client.hkeys(self.penalties_key))

        log_data = {
            "organic_history": original_history_count - current_history_count,
            "organic_penalties": original_penalties_count - current_penalties_count,
        }

        bt.logging.info(
            f"Removed deregistered hotkeys from organic query state: {log_data}"
        )

    def save_organic_query_history(
        self, hotkey: str, synapse: Any, is_failed: bool
    ) -> None:
        """Save a synapse and its failed state to the history"""
        # Get current history for this hotkey
        history_json = redis_client.hget(self.history_key, hotkey)
        history = json.loads(history_json) if history_json else []

        # Serialize the synapse object
        serialized_synapse = jsonpickle.encode(synapse)

        # Add the new entry
        history.append([serialized_synapse, is_failed])

        # Store back to Redis
        redis_client.hset(self.history_key, hotkey, json.dumps(history))

    def collect_failed_synapses(self) -> List[Tuple[str, Any]]:
        """Collect all failed synapses from history"""
        failed_synapses = []

        # Get all hotkeys and their history
        all_histories = redis_client.hgetall(self.history_key)

        for hotkey, history_json in all_histories.items():
            history = json.loads(history_json)

            # Extract failed synapses
            for serialized_synapse, is_failed in history:
                if is_failed:
                    synapse = jsonpickle.decode(serialized_synapse)
                    failed_synapses.append((hotkey, synapse))

        # If there are no failed synapses, collect all synapses
        if not failed_synapses:
            for hotkey, history_json in all_histories.items():
                history = json.loads(history_json)

                for serialized_synapse, _ in history:
                    synapse = jsonpickle.decode(serialized_synapse)
                    failed_synapses.append((hotkey, synapse))

        return failed_synapses

    def parse_datetime(self, date_str: str, format_str: str) -> datetime:
        """Parse a datetime string into a datetime object with UTC timezone"""
        return datetime.strptime(date_str, format_str).replace(tzinfo=pytz.utc)

    def get_specified_uids(self, uids, synapse_uid) -> List[int]:
        """Get all uids except the one that made the query"""
        return [uid for uid in uids if uid != synapse_uid]

    def clear_history(self) -> None:
        """Clear the organic history"""
        redis_client.delete(self.history_key)
