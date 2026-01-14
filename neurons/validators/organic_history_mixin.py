import random
import time
from typing import Dict, List

import jsonpickle
import torch

from desearch.redis.redis_client import redis_client


class OrganicHistoryMixin:
    HISTORY_EXPIRY_TIME = 2 * 3600  # 2 hours in seconds

    def __init__(self):
        pass

    def _get_uid_key(self, uid: int) -> str:
        """Get Redis key for a specific UID"""
        return f"{self.__class__.__name__}:organic_history:uid:{uid}"

    def _get_all_keys_pattern(self) -> str:
        """Get pattern to match all UID keys"""
        return f"{self.__class__.__name__}:organic_history:uid:*"

    async def _save_organic_response(
        self, uids, responses, tasks, event, start_time
    ) -> None:
        """Save organic responses to Redis using sorted sets"""
        pipeline = redis_client.pipeline()

        for uid, response, task, *event_values in zip(
            uids, responses, tasks, *event.values()
        ):
            event_dict = dict(zip(event.keys(), event_values))

            # Prepare data to store
            data = {
                "response": response,
                "task": task,
                "event": event_dict,
                "start_time": start_time,
            }

            # Use ZADD with timestamp as score
            # Use jsonpickle for serialization to handle Pydantic models
            key = self._get_uid_key(uid.item())
            pipeline.zadd(key, {jsonpickle.encode(data): start_time})

            # Set key expiry to slightly longer than HISTORY_EXPIRY_TIME
            # This ensures Redis eventually cleans up abandoned keys
            pipeline.expire(key, self.HISTORY_EXPIRY_TIME + 3600)

        await pipeline.execute()

    async def _clean_uid_history(self, uid: int) -> int:
        """Clean expired entries for a specific UID and delete key if empty"""
        key = self._get_uid_key(uid)
        current_time = time.time()
        cutoff_time = current_time - self.HISTORY_EXPIRY_TIME

        # Remove old entries
        removed_count = await redis_client.zremrangebyscore(key, "-inf", cutoff_time)

        # Check if sorted set is empty and delete if so
        if await redis_client.zcard(key) == 0:
            await redis_client.delete(key)

        return removed_count

    async def _clean_all_history(self) -> Dict[int, int]:
        """Clean expired entries for all UIDs"""
        # Get all UID keys
        keys_pattern = self._get_all_keys_pattern()
        keys = await redis_client.keys(keys_pattern)

        if not keys:
            return {}

        # Extract UIDs from keys
        removed_counts = {}
        for key in keys:
            # Extract UID from key format
            uid = int(key.split(":")[-1])
            removed_count = await self._clean_uid_history(uid)
            if removed_count > 0:
                removed_counts[uid] = removed_count

        return removed_counts

    async def get_random_organic_responses(self):
        """Get random organic responses from all UIDs with history"""
        # Clean all history first
        await self._clean_all_history()

        # Get all UID keys that still exist
        keys = await redis_client.keys(self._get_all_keys_pattern())

        if not keys:
            return {
                "event": {},
                "tasks": [],
                "responses": [],
                "uids": torch.tensor([]),
            }

        event = {}
        tasks = []
        responses = []
        uids = []

        # Use pipeline for efficiency
        pipeline = redis_client.pipeline()
        for key in keys:
            pipeline.zrange(key, 0, -1)  # Get all members

        all_histories = await pipeline.execute()

        for key, history_data in zip(keys, all_histories):
            if not history_data:  # Skip if no data
                continue

            # Extract UID from key
            uid = int(key.split(":")[-1])
            uids.append(torch.tensor([uid]))

            # Pick random entry
            random_entry = random.choice(history_data)
            data = jsonpickle.decode(random_entry)

            responses.append(data["response"])
            tasks.append(data["task"])

            # Aggregate event data
            for key, value in data["event"].items():
                if key not in event:
                    event[key] = []
                event[key].append(value)

        return {
            "event": event,
            "tasks": tasks,
            "responses": responses,
            "uids": torch.tensor(uids) if uids else torch.tensor([]),
        }

    async def get_uids_with_no_history(self, available_uids: List[int]) -> List[int]:
        """Get UIDs that have no history in Redis"""
        # Clean all history first
        await self._clean_all_history()

        # Check which UIDs have keys in Redis
        pipeline = redis_client.pipeline()
        for uid in available_uids:
            pipeline.exists(self._get_uid_key(uid))

        exists_results = await pipeline.execute()

        # Return UIDs where key doesn't exist
        return [
            uid for uid, exists in zip(available_uids, exists_results) if not exists
        ]
