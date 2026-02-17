import asyncio
import logging
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import bittensor as bt
from fastapi import HTTPException
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.domains.dataset.enums import SearchType
from app.domains.dataset.models.question import Question
from app.domains.dataset.schemas import QuestionOut

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


@dataclass
class EpochCache:
    """Holds the cached question assignments for a single epoch."""

    epoch_id: int = -1
    miner_uids: List[int] = field(default_factory=list)
    # Deterministic mapping: search_type -> {uid: question}
    assignments: Dict[SearchType, Dict[int, QuestionOut]] = field(default_factory=dict)


class EpochQuestionCache:
    """
    Manages per-epoch question caching with per-validator serving state.

    Each epoch, questions are assigned deterministically to (uid, search_type)
    pairs. All validators receive the same question for the same pair, but in
    random order.  The cache tracks which pairs have been served to each
    validator so duplicates are never returned.
    """

    def __init__(self, netuid: int, subtensor_network: str):
        self.netuid = netuid
        self.subtensor_network = subtensor_network
        self._cache = EpochCache()
        self._lock = asyncio.Lock()
        self._subtensor: Optional[bt.AsyncSubtensor] = None
        self._tempo: Optional[int] = None

        # Track served (search_type, uid) per validator hotkey
        self._served: Dict[str, Set[Tuple[SearchType, int]]] = {}

    async def initialize(self):
        """Connect to subtensor and fetch tempo."""
        self._subtensor = bt.AsyncSubtensor(network=self.subtensor_network)
        await self._subtensor.initialize()
        current_block = await self._subtensor.get_current_block()
        self._tempo = await self._subtensor.tempo(self.netuid, current_block)
        logger.info(
            f"EpochQuestionCache initialized: netuid={self.netuid}, "
            f"tempo={self._tempo}, block={current_block}"
        )

    async def close(self):
        """Clean up subtensor connection."""
        if self._subtensor:
            await self._subtensor.close()

    def _compute_epoch_id(self, block: int) -> int:
        return block // self._tempo

    async def _refresh_cache(self, session: AsyncSession):
        """Fetch metagraph for miner UIDs and build question assignments."""
        current_block = await self._subtensor.get_current_block()
        epoch_id = self._compute_epoch_id(current_block)

        # Refresh tempo in case it changed
        self._tempo = await self._subtensor.tempo(self.netuid, current_block)

        # Get all UIDs from metagraph
        metagraph = await self._subtensor.metagraph(self.netuid)
        miner_uids = [uid for uid in metagraph.uids]

        logger.info(
            f"Refreshing epoch cache: epoch_id={epoch_id}, "
            f"uid_count={len(miner_uids)}, block={current_block}"
        )

        # Build deterministic assignments per search type
        assignments: Dict[SearchType, Dict[int, QuestionOut]] = {}

        for search_type in SearchType:
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

            # uid[i] -> question[i % len(questions)]
            assignments[search_type] = {
                uid: questions[i % len(questions)] for i, uid in enumerate(miner_uids)
            }

            logger.info(
                f"Assigned {len(assignments[search_type])} questions "
                f"for {search_type.value} (epoch {epoch_id})"
            )

        self._cache = EpochCache(
            epoch_id=epoch_id,
            miner_uids=miner_uids,
            assignments=assignments,
        )

        # Clear serving state on epoch change
        self._served.clear()

    async def _ensure_fresh(self, session: AsyncSession):
        """Refresh cache if epoch has changed."""
        current_block = await self._subtensor.get_current_block()
        current_epoch = self._compute_epoch_id(current_block)

        if current_epoch != self._cache.epoch_id:
            async with self._lock:
                fresh_block = await self._subtensor.get_current_block()
                fresh_epoch = self._compute_epoch_id(fresh_block)
                if fresh_epoch != self._cache.epoch_id:
                    await self._refresh_cache(session)

    async def get_next_question(
        self, session: AsyncSession, hotkey: str
    ) -> Tuple[int, int, SearchType, QuestionOut]:
        """
        Return one random unserved (uid, search_type, question) for this
        validator. All validators get the same question for the same
        (uid, search_type) pair.

        Raises HTTPException 404 when all questions have been served.
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
                detail="All questions served for this epoch",
            )

        # Pick random unserved combo
        search_type, uid = random.choice(unserved)
        served.add((search_type, uid))

        question = self._cache.assignments[search_type][uid]
        return self._cache.epoch_id, uid, search_type, question
