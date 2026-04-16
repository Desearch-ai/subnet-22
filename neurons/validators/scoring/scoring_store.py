from datetime import datetime
from typing import Any, Dict, List

import jsonpickle

from desearch.redis.redis_client import redis_client

EXPIRY = 2 * 3600  # 2 hours

SEARCH_TYPES = ["ai_search", "x_search", "web_search"]


class ScoringStore:
    """
    Redis-backed store for scoring query responses.

    Organizes responses by (time_range_start, search_type) using Redis hashes.
    Key format: scoring:{unix_ts}:{search_type}
    Field:      {uid}  →  jsonpickle-encoded response

    Responses expire after 2 hours.
    """

    KEY_PREFIX = "scoring"

    def _key(self, time_range_start: datetime, search_type: str) -> str:
        unix_ts = int(time_range_start.timestamp())
        return f"{self.KEY_PREFIX}:{unix_ts}:{search_type}"

    async def save_response(
        self,
        time_range_start: datetime,
        uid: int,
        search_type: str,
        response: Any,
        scoring_seed: int | None = None,
    ) -> None:
        """Save a single miner response (with optional scoring seed) for later scoring."""

        key = self._key(time_range_start, search_type)
        data = jsonpickle.encode({"response": response, "scoring_seed": scoring_seed})
        pipeline = redis_client.pipeline()
        pipeline.hset(key, str(uid), data)
        pipeline.expire(key, EXPIRY)
        await pipeline.execute()

    async def get_all_for_range(
        self, time_range_start: datetime
    ) -> Dict[str, List[Dict]]:
        """
        Load all saved responses for a completed epoch.

        Returns:
            {
                "ai_search": [{"uid": int, "response": ...}, ...],
                "x_search":  [...],
                "web_search": [...],
            }
        """

        pipeline = redis_client.pipeline()

        for st in SEARCH_TYPES:
            pipeline.hgetall(self._key(time_range_start, st))

        raw_results = await pipeline.execute()

        result: Dict[str, List[Dict]] = {}

        for st, raw in zip(SEARCH_TYPES, raw_results):
            items = []

            for uid_str, encoded in raw.items():
                data = jsonpickle.decode(encoded)
                if isinstance(data, dict) and "response" in data:
                    response = data["response"]
                    scoring_seed = data.get("scoring_seed")
                else:
                    response = data
                    scoring_seed = None
                items.append(
                    {
                        "uid": int(uid_str),
                        "response": response,
                        "scoring_seed": scoring_seed,
                    }
                )

            if items:
                result[st] = items

        return result
