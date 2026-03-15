import asyncio
import logging
import random
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Set, Tuple

import bittensor as bt
from app.domains.dataset.enums import SearchType
from app.domains.dataset.models.question import Question
from app.domains.dataset.schemas import QuestionOut
from fastapi import HTTPException
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

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

    if selected_field == "date_range":
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


def _current_hour_utc() -> datetime:
    """Return the start of the current UTC hour (wall-clock aligned)."""
    now = datetime.now(timezone.utc)
    return now.replace(minute=0, second=0, microsecond=0)


@dataclass
class TimeRangeCache:
    """Holds the cached question assignments for a single hourly time range."""

    time_range_start: Optional[datetime] = None
    miner_uids: List[int] = field(default_factory=list)
    # Deterministic mapping: search_type -> {uid: question} (params embedded in QuestionOut)
    assignments: Dict[SearchType, Dict[int, QuestionOut]] = field(default_factory=dict)


class QuestionCache:
    """
    Manages per-hour question caching with per-validator serving state.

    Each UTC hour, questions are assigned deterministically to (uid, search_type)
    pairs. All validators receive the same question for the same pair, but in
    random order. The cache tracks which pairs have been served to each
    validator so duplicates are never returned within the hour.

    At the start of a new hour the cache is refreshed lazily on the next request.
    """

    def __init__(self, netuid: int, subtensor_network: str):
        self.netuid = netuid
        self.subtensor_network = subtensor_network
        self._cache = TimeRangeCache()
        self._lock = asyncio.Lock()
        self._subtensor: Optional[bt.AsyncSubtensor] = None

        # Track served (search_type, uid) per validator hotkey
        self._served: Dict[str, Set[Tuple[SearchType, int]]] = {}

    async def initialize(self):
        """Connect to subtensor."""
        self._subtensor = bt.AsyncSubtensor(network=self.subtensor_network)
        await self._subtensor.initialize()
        current_block = await self._subtensor.get_current_block()
        logger.info(
            f"QuestionCache initialized: netuid={self.netuid}, block={current_block}"
        )

    async def close(self):
        """Clean up subtensor connection."""
        if self._subtensor:
            await self._subtensor.close()

    async def _refresh_cache(self, session: AsyncSession):
        """Fetch metagraph for miner UIDs and build question assignments."""
        time_range_start = _current_hour_utc()

        # Get all UIDs from metagraph
        metagraph = await self._subtensor.metagraph(self.netuid)
        miner_uids = [uid for uid in metagraph.uids]

        logger.info(
            f"Refreshing question cache: time_range={time_range_start.isoformat()}, "
            f"uid_count={len(miner_uids)}"
        )

        # Build deterministic assignments per search type
        assignments: Dict[SearchType, Dict[int, QuestionOut]] = {}

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
                logger.warning(f"No questions found for {search_type.value}")
                assignments[search_type] = {}
                continue

            # uid[i] -> question[i % len(questions)] with per-uid params embedded
            assignments[search_type] = {
                uid: QuestionOut(
                    query=questions[i % len(questions)].query,
                    params=_generate_params_for(search_type),
                )
                for i, uid in enumerate(miner_uids)
            }

            logger.info(
                f"Assigned {len(assignments[search_type])} questions "
                f"for {search_type.value} (time_range {time_range_start.isoformat()})"
            )

        self._cache = TimeRangeCache(
            time_range_start=time_range_start,
            miner_uids=miner_uids,
            assignments=assignments,
        )

        # Clear serving state on time range change
        self._served.clear()

    async def _ensure_fresh(self, session: AsyncSession):
        """Refresh cache if the current UTC hour differs from the cached one."""
        if _current_hour_utc() != self._cache.time_range_start:
            async with self._lock:
                # Re-check after acquiring lock to avoid double refresh
                if _current_hour_utc() != self._cache.time_range_start:
                    await self._refresh_cache(session)

    async def get_next_question(
        self, session: AsyncSession, hotkey: str
    ) -> Tuple[datetime, int, SearchType, QuestionOut]:
        """
        Return one random unserved (uid, search_type, question) for this validator.
        Params are embedded inside the returned QuestionOut. All validators get the
        same question and params for the same (uid, search_type) pair within the
        current UTC hour.

        Raises HTTPException 404 when all questions have been served for the hour.
        """
        await self._ensure_fresh(session)

        served = self._served.setdefault(hotkey, set())

        # Build list of unserved combos
        unserved = []
        for search_type, uid_map in self._cache.assignments.items():
            for uid in uid_map:
                if (search_type, uid) not in served:
                    unserved.append((search_type, uid))

        if not unserved:
            raise HTTPException(
                status_code=404,
                detail="All questions served for this time range",
            )

        # Pick random unserved combo
        search_type, uid = random.choice(unserved)
        served.add((search_type, uid))

        question = self._cache.assignments[search_type][uid]
        return self._cache.time_range_start, uid, search_type, question
