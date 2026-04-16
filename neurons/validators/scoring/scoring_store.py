from datetime import datetime
from typing import Any, Dict, List
from uuid import uuid4

import jsonpickle

from desearch.redis.redis_client import redis_client

EXPIRY = 2 * 3600  # 2 hours

SEARCH_TYPES = ["ai_search", "x_search", "web_search"]


class ScoringStore:
    """
    Redis-backed store for scoring responses (synthetic + organic).

    Keys:
        scoring:{unix_ts}:synthetic:{search_type}
        scoring:{unix_ts}:organic:{search_type}

    Field layout inside each hash: {uid}:{suffix} → jsonpickle-encoded response.
    Multiple responses per UID are supported; all entries expire after 2h.
    """

    KEY_PREFIX = "scoring"

    def _key(self, time_range_start: datetime, kind: str, search_type: str) -> str:
        unix_ts = int(time_range_start.timestamp())
        return f"{self.KEY_PREFIX}:{unix_ts}:{kind}:{search_type}"

    async def _save(
        self,
        time_range_start: datetime,
        kind: str,
        uid: int,
        search_type: str,
        response: Any,
    ) -> None:
        key = self._key(time_range_start, kind, search_type)
        field = f"{uid}:{uuid4().hex[:8]}"
        data = jsonpickle.encode(response)
        pipeline = redis_client.pipeline()
        pipeline.hset(key, field, data)
        pipeline.expire(key, EXPIRY)
        await pipeline.execute()

    async def save_synthetic(
        self,
        time_range_start: datetime,
        uid: int,
        search_type: str,
        response: Any,
    ) -> None:
        await self._save(time_range_start, "synthetic", uid, search_type, response)

    async def save_organic(
        self,
        time_range_start: datetime,
        uid: int,
        search_type: str,
        response: Any,
    ) -> None:
        await self._save(time_range_start, "organic", uid, search_type, response)

    async def _load(
        self, time_range_start: datetime, kind: str
    ) -> Dict[str, List[Dict]]:
        pipeline = redis_client.pipeline()

        for st in SEARCH_TYPES:
            pipeline.hgetall(self._key(time_range_start, kind, st))

        raw_results = await pipeline.execute()

        result: Dict[str, List[Dict]] = {}

        for st, raw in zip(SEARCH_TYPES, raw_results):
            items = []
            for field_str, encoded in raw.items():
                uid_part = field_str.split(":")[0] if ":" in field_str else field_str
                response = jsonpickle.decode(encoded)
                items.append({"uid": int(uid_part), "response": response})
            if items:
                result[st] = items

        return result

    async def get_synthetics_for_range(
        self, time_range_start: datetime
    ) -> Dict[str, List[Dict]]:
        return await self._load(time_range_start, "synthetic")

    async def get_organics_for_range(
        self, time_range_start: datetime
    ) -> Dict[str, List[Dict]]:
        return await self._load(time_range_start, "organic")
