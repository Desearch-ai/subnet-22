import asyncio
import hashlib
import json
import random
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Set, Tuple

import bittensor as bt
from app.domains.dataset.enums import SearchType
from app.domains.dataset.models.question import Question
from app.domains.dataset.schemas import QuestionOut
from app.logger import get_logger
from app.redis_client import get_redis
from fastapi import HTTPException
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

logger = get_logger(__name__)

_KEY_UIDS = "qcache:uids:{hour}"
_KEY_ASSIGN = "qcache:assign:{hour}:{search_type}"
_KEY_SERVED = "qcache:served:{hour}:{hotkey}"
_KEY_ACTIVE = "scoring:active:{hour}"
_TTL = 7200

_AI_SEARCH_TOOLS: List[List[str]] = [
    ["Twitter Search", "Reddit Search"],
    ["Twitter Search", "Web Search"],
    ["Twitter Search", "Web Search"],
    ["Twitter Search", "Web Search"],
    ["Twitter Search", "Web Search"],
    ["Twitter Search", "Hacker News Search"],
    ["Twitter Search", "Hacker News Search"],
    ["Twitter Search", "Youtube Search"],
    ["Twitter Search", "Youtube Search"],
    ["Twitter Search", "Youtube Search"],
    ["Twitter Search", "Web Search"],
    ["Twitter Search", "Reddit Search"],
    ["Twitter Search", "Reddit Search"],
    ["Twitter Search", "Hacker News Search"],
    ["Twitter Search", "ArXiv Search"],
    ["Twitter Search", "ArXiv Search"],
    ["Twitter Search", "Wikipedia Search"],
    ["Twitter Search", "Wikipedia Search"],
    ["Twitter Search", "Web Search"],
    ["Twitter Search", "Web Search"],
    ["Twitter Search", "Web Search"],
    ["Web Search"],
    ["Reddit Search"],
    ["Hacker News Search"],
    ["Youtube Search"],
    ["ArXiv Search"],
    ["Wikipedia Search"],
    ["Twitter Search", "Youtube Search", "ArXiv Search", "Wikipedia Search"],
    ["Twitter Search", "Web Search", "Reddit Search", "Hacker News Search"],
    [
        "Twitter Search",
        "Web Search",
        "Reddit Search",
        "Hacker News Search",
        "Youtube Search",
        "ArXiv Search",
        "Wikipedia Search",
    ],
]

_AI_SEARCH_DATE_FILTERS: List[str] = list(
    Counter(
        {
            "PAST_24_HOURS": 4,
            "PAST_2_DAYS": 5,
            "PAST_WEEK": 5,
            "PAST_2_WEEKS": 5,
            "PAST_MONTH": 1,
            "PAST_YEAR": 1,
        }
    ).elements()
)

_X_SEARCH_PARAM_FIELDS: List[str] = [
    "sort",
    "is_quote",
    "is_video",
    "is_image",
    "min_retweets",
    "min_replies",
    "min_likes",
    "date_range",
]

_THREE_YEARS_IN_DAYS = 3 * 365
_SCORING_SEARCH_TYPES: Tuple[SearchType, ...] = (
    SearchType.AI_SEARCH,
    SearchType.X_SEARCH,
    SearchType.WEB_SEARCH,
)


def _generate_ai_search_params() -> Dict[str, Any]:
    return {
        "tools": random.choice(_AI_SEARCH_TOOLS),
        "date_filter_type": random.choice(_AI_SEARCH_DATE_FILTERS),
    }


def _generate_x_search_params() -> Dict[str, Any]:
    selected_field = random.choice(_X_SEARCH_PARAM_FIELDS)
    params: Dict[str, Any] = {}

    if selected_field == "sort":
        params["sort"] = "Latest"
    elif selected_field == "date_range":
        now = datetime.now(timezone.utc)
        end_date = now - timedelta(days=random.randint(0, _THREE_YEARS_IN_DAYS))
        start_date = end_date - timedelta(days=random.randint(7, 14))
        params["start_date"] = start_date.strftime("%Y-%m-%d_%H:%M:%S_UTC")
        params["end_date"] = end_date.strftime("%Y-%m-%d_%H:%M:%S_UTC")
    elif selected_field == "is_video":
        params["is_video"] = random.choice([True, False])
    elif selected_field == "is_image":
        params["is_image"] = random.choice([True, False])
    elif selected_field == "is_quote":
        params["is_quote"] = random.choice([True, False])
    elif selected_field == "min_likes":
        params["min_likes"] = random.randint(5, 100)
    elif selected_field == "min_replies":
        params["min_replies"] = random.randint(5, 20)
    elif selected_field == "min_retweets":
        params["min_retweets"] = random.randint(5, 20)

    return params


def _generate_params_for(search_type: "SearchType") -> Dict[str, Any]:
    if search_type == SearchType.AI_SEARCH:
        return _generate_ai_search_params()
    if search_type == SearchType.X_SEARCH:
        return _generate_x_search_params()
    return {}  # web_search — no extra params


def compute_scoring_fingerprint(search_type: str, query: str) -> str:
    blob = json.dumps({"q": query, "s": search_type}, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()


def _current_hour_utc() -> datetime:
    """Return the start of the current UTC hour (wall-clock aligned)."""
    now = datetime.now(timezone.utc)
    return now.replace(minute=0, second=0, microsecond=0)


def _generate_random_seed() -> int:
    return random.SystemRandom().randint(0, 0x7FFFFFFF)


def _hour_key(dt: datetime) -> str:
    """ISO-format key for Redis, e.g. '2026-04-10T14:00:00+00:00'."""
    return dt.isoformat()


@dataclass
class TimeRangeCache:
    """Holds the cached question assignments for a single hourly time range."""

    time_range_start: Optional[datetime] = None
    miner_uids: List[int] = field(default_factory=list)
    assignments: Dict[SearchType, Dict[int, QuestionOut]] = field(default_factory=dict)

    # Shared scoring seeds for the current cache instance: search_type -> {uid: seed}
    scoring_seeds: Dict[SearchType, Dict[int, int]] = field(default_factory=dict)


class QuestionCache:
    """
    Manages per-hour question caching with per-validator serving state.

    Redis-backed so that redeployments within the same hour do not reset state.

    Each UTC hour, questions are assigned deterministically to (uid, search_type)
    pairs. All validators receive the same question for the same pair, but in
    random order. The cache tracks which pairs have been served to each
    validator so duplicates are never returned within the hour.
    """

    def __init__(self, netuid: int, subtensor_network: str):
        self.netuid = netuid
        self.subtensor_network = subtensor_network
        self._cache = TimeRangeCache()
        self._lock = asyncio.Lock()
        self._subtensor: Optional[bt.AsyncSubtensor] = None

    async def initialize(self):
        """Connect to subtensor."""
        self._subtensor = bt.AsyncSubtensor(network=self.subtensor_network)
        await self._subtensor.initialize()
        current_block = await self._subtensor.get_current_block()
        logger.info(
            f"QuestionCache initialized: netuid={self.netuid} "
            f"subtensor_network={self.subtensor_network} "
            f"block={current_block}"
        )

    async def close(self):
        """Clean up subtensor connection."""
        if self._subtensor:
            await self._subtensor.close()

    async def _save_to_redis(self) -> None:
        """Persist current in-memory cache to Redis (pipeline for atomicity)."""
        r = get_redis()
        hour = _hour_key(self._cache.time_range_start)
        pipe = r.pipeline()

        # Miner UIDs
        pipe.set(
            _KEY_UIDS.format(hour=hour),
            json.dumps(self._cache.miner_uids),
            ex=_TTL,
        )

        fingerprints: list[str] = []
        for search_type, uid_map in self._cache.assignments.items():
            hash_key = _KEY_ASSIGN.format(hour=hour, search_type=search_type.value)
            for uid, question in uid_map.items():
                pipe.hset(hash_key, str(uid), question.model_dump_json())
                fp = compute_scoring_fingerprint(search_type.value, question.query)
                fingerprints.append(fp)
            pipe.expire(hash_key, _TTL)

        if fingerprints:
            active_key = _KEY_ACTIVE.format(hour=hour)
            pipe.sadd(active_key, *fingerprints)
            pipe.expire(active_key, _TTL)

        await pipe.execute()
        logger.info(f"Saved cache to Redis: hour={hour}")

    async def _load_from_redis(self, hour_dt: datetime) -> bool:
        """
        Try to load assignments for *hour_dt* from Redis.

        Returns True if Redis had a complete cache for this hour,
        False otherwise (caller should regenerate from the DB).
        """
        r = get_redis()
        hour = _hour_key(hour_dt)

        uids_json = await r.get(_KEY_UIDS.format(hour=hour))
        if not uids_json:
            return False

        miner_uids: List[int] = json.loads(uids_json)
        assignments: Dict[SearchType, Dict[int, QuestionOut]] = {}

        for search_type in _SCORING_SEARCH_TYPES:
            hash_key = _KEY_ASSIGN.format(hour=hour, search_type=search_type.value)
            raw = await r.hgetall(hash_key)
            if raw:
                assignments[search_type] = {
                    int(uid_str): QuestionOut.model_validate_json(q_json)
                    for uid_str, q_json in raw.items()
                }
            else:
                assignments[search_type] = {}

        self._cache = TimeRangeCache(
            time_range_start=hour_dt,
            miner_uids=miner_uids,
            assignments=assignments,
        )

        total = sum(len(m) for m in assignments.values())
        logger.info(
            f"Loaded cache from Redis: hour={hour} "
            f"uids={len(miner_uids)} total_assignments={total}"
        )
        return True

    async def _get_served(self, hour: str, hotkey: str) -> Set[str]:
        """Return the set of 'search_type:uid' strings already served."""
        r = get_redis()
        return await r.smembers(_KEY_SERVED.format(hour=hour, hotkey=hotkey))

    async def _mark_served(
        self, hour: str, hotkey: str, search_type: str, uid: int
    ) -> None:
        """Record that (search_type, uid) was served to this validator."""
        r = get_redis()
        key = _KEY_SERVED.format(hour=hour, hotkey=hotkey)
        await r.sadd(key, f"{search_type}:{uid}")
        await r.expire(key, _TTL)

    def _assignment_counts(self) -> dict[str, int]:
        return {
            search_type.value: len(uid_map)
            for search_type, uid_map in self._cache.assignments.items()
        }

    async def _refresh_cache(self, session: AsyncSession):
        """Fetch metagraph for miner UIDs and build question assignments."""
        time_range_start = _current_hour_utc()
        previous_time_range = self._cache.time_range_start

        logger.info(
            "Refreshing question cache: "
            f"previous_time_range="
            f"{previous_time_range.isoformat() if previous_time_range else None} "
            f"next_time_range={time_range_start.isoformat()}"
        )

        try:
            # Get all UIDs from metagraph
            metagraph = await self._subtensor.metagraph(self.netuid)
            miner_uids = [int(uid) for uid in metagraph.uids]

            logger.info(
                f"Fetched metagraph for refresh: "
                f"time_range={time_range_start.isoformat()} "
                f"uid_count={len(miner_uids)}"
            )

            # Build deterministic assignments and per-cache scoring seeds per search type
            assignments: Dict[SearchType, Dict[int, QuestionOut]] = {}
            scoring_seeds: Dict[SearchType, Dict[int, int]] = {}

            for search_type in _SCORING_SEARCH_TYPES:
                stmt = (
                    select(Question)
                    .where(Question.search_types.contains([search_type.value]))
                    .order_by(func.random())
                    .limit(len(miner_uids))
                )
                result = await session.execute(stmt)
                rows = result.scalars().all()
                questions = [QuestionOut.model_validate(q) for q in rows]

                if not questions:
                    logger.warning(
                        f"No questions found for search_type={search_type.value} "
                        f"time_range={time_range_start.isoformat()}"
                    )
                    assignments[search_type] = {}
                    scoring_seeds[search_type] = {}
                    continue

                # AI search params should be identical for every miner within the
                # scoring window, while other search types keep their current
                # per-miner generation behavior.
                shared_params = (
                    _generate_params_for(search_type)
                    if search_type == SearchType.AI_SEARCH
                    else None
                )

                uid_questions = {}
                uid_seeds = {}
                for i, uid in enumerate(miner_uids):
                    uid_questions[uid] = QuestionOut(
                        query=questions[i % len(questions)].query,
                        params=(
                            dict(shared_params)
                            if shared_params is not None
                            else _generate_params_for(search_type)
                        ),
                    )
                    uid_seeds[uid] = _generate_random_seed()

                assignments[search_type] = uid_questions
                scoring_seeds[search_type] = uid_seeds

                sample_question = questions[0].query if questions else ""
                logger.info(
                    f"Assigned questions: "
                    f"time_range={time_range_start.isoformat()} "
                    f"search_type={search_type.value} "
                    f"assignment_count={len(assignments[search_type])} "
                    f"sampled_questions={len(questions)} "
                    f"shared_params={shared_params} "
                    f"sample_query={sample_question[:120]!r}"
                )

            self._cache = TimeRangeCache(
                time_range_start=time_range_start,
                miner_uids=miner_uids,
                assignments=assignments,
                scoring_seeds=scoring_seeds,
            )

            await self._save_to_redis()

            logger.info(
                f"Question cache refresh complete: "
                f"time_range={time_range_start.isoformat()} "
                f"assignments={self._assignment_counts()}"
            )
        except Exception:
            logger.exception(
                "Question cache refresh failed: "
                f"previous_time_range="
                f"{previous_time_range.isoformat() if previous_time_range else None} "
                f"next_time_range={time_range_start.isoformat()}"
            )
            raise

    async def _ensure_fresh(self, session: AsyncSession):
        """Refresh cache if the current UTC hour differs from the cached one."""
        current_hour = _current_hour_utc()
        if current_hour != self._cache.time_range_start:
            logger.info(
                f"Detected cache rollover: "
                f"cached_time_range="
                f"{self._cache.time_range_start.isoformat() if self._cache.time_range_start else None} "
                f"current_hour={current_hour.isoformat()}"
            )
            async with self._lock:
                # Re-check after acquiring lock to avoid double refresh
                current_hour = _current_hour_utc()
                if current_hour != self._cache.time_range_start:
                    loaded = await self._load_from_redis(current_hour)
                    if not loaded:
                        await self._refresh_cache(session)

    async def get_next_question(
        self, session: AsyncSession, hotkey: str
    ) -> Tuple[datetime, int, SearchType, QuestionOut, int]:
        """
        Return one random unserved (uid, search_type, question, scoring_seed) for
        this validator. Params are embedded inside the returned QuestionOut. All
        validators get the same question, params, and scoring_seed for the same
        (uid, search_type) pair for the current cache window.

        The scoring_seed is generated when the hourly cache is refreshed and
        should be used by validators to seed random selection of tweets/links
        for validation, ensuring consistent scoring across validators while the
        cache remains in memory.

        Raises HTTPException 404 when all questions have been served for the hour.
        """
        await self._ensure_fresh(session)

        hour = _hour_key(self._cache.time_range_start)
        served = await self._get_served(hour, hotkey)

        # Build list of unserved combos
        unserved = []
        for search_type, uid_map in self._cache.assignments.items():
            for uid in uid_map:
                member = f"{search_type.value}:{uid}"
                if member not in served:
                    unserved.append((search_type, uid))

        if not unserved:
            logger.info(
                f"All questions served for validator: hotkey={hotkey} "
                f"time_range={hour} "
                f"served_count={len(served)} "
                f"assignments={self._assignment_counts()}"
            )
            raise HTTPException(
                status_code=404,
                detail="All questions served for this time range",
            )

        # Pick random unserved combo
        search_type, uid = random.choice(unserved)

        await self._mark_served(hour, hotkey, search_type.value, uid)

        question = self._cache.assignments[search_type][uid]

        scoring_seed = self._cache.scoring_seeds.get(search_type, {}).get(uid, 0)

        logger.debug(
            f"Serving question: hotkey={hotkey} "
            f"time_range={hour} "
            f"search_type={search_type.value} "
            f"uid={uid} "
            f"served_count={len(served) + 1} "
            f"remaining={len(unserved) - 1} "
            f"params={question.params} "
            f"scoring_seed={scoring_seed} "
            f"query={question.query[:120]!r}"
        )
        return self._cache.time_range_start, uid, search_type, question, scoring_seed
