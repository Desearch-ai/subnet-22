import asyncio
import random
from typing import List

import bittensor as bt

from desearch.dataset import BasicQuestionsDataset, QuestionsDataset
from desearch.dataset.date_filters import random_date_filters
from desearch.dataset.hf_dataset import HFQuestionPool
from desearch.protocol import ScoringModel
from desearch.utils import (
    AI_SEARCH_MODES,
    SearchMode,
    get_mode_serving_budget,
)

WEB_TOOL = "Web Search"
TWITTER_TOOL = "Twitter Search"

SEARCH_TYPES = ["ai_search", "x_search", "web_search"]

X_LANE = "x"
WEB_LANES = ("news", "squad", "nq")


def pick_ai_mode_and_tool() -> tuple[SearchMode, list[str]]:
    mode = random.choice(AI_SEARCH_MODES)
    tool = random.choice([WEB_TOOL, TWITTER_TOOL])
    return mode, [tool]


class SyntheticQueryGenerator:
    """
    Generates synthetic scoring queries locally using the existing
    desearch/dataset module + OpenAI for question enhancement.
    """

    MAX_CONCURRENT_LLM = 20  # Throttle concurrent OpenAI calls

    def __init__(self):
        self.questions_dataset = QuestionsDataset()
        self.basic_dataset = BasicQuestionsDataset()
        self.hf_pool = HFQuestionPool()

    async def generate_epoch_queries(
        self,
        available_uids: List[int],
        verified_by_type: dict[str, dict[int, int]] | None = None,
        scoring_model: ScoringModel = ScoringModel.OPENAI_GPT4_1_NANO,
    ) -> List[dict]:
        if verified_by_type is None:
            verified_by_type = {}

        ai_date_filter = random.choice(random_date_filters)

        items = await asyncio.to_thread(
            self._generate_dataset_queries, available_uids, verified_by_type
        )
        if items is not None:
            return items
        bt.logging.warning(
            "[SyntheticGen] Dataset pool unavailable, falling back to LLM path"
        )

        bt.logging.info(
            f"[SyntheticGen] Epoch params: date_filter={ai_date_filter.value}"
        )

        items: List[dict] = []
        llm_items: List[dict] = []  # Only ai_search + web_search need LLM

        for uid in available_uids:
            for search_type in SEARCH_TYPES:
                n = verified_by_type.get(search_type, {}).get(uid, 1)
                for _ in range(n):
                    item = {
                        "uid": uid,
                        "search_type": search_type,
                        "query": None,
                    }

                    if search_type == "x_search":
                        item["query"] = {
                            "query": self.basic_dataset.generate_random_x_query()
                        }
                    else:
                        llm_items.append(item)

                    items.append(item)

        semaphore = asyncio.Semaphore(self.MAX_CONCURRENT_LLM)

        async def _generate_one(item: dict) -> None:
            async with semaphore:
                try:
                    if item["search_type"] == "ai_search":
                        mode, ai_tools = pick_ai_mode_and_tool()
                        question = await self.questions_dataset.generate_new_question_with_openai(
                            ai_tools, model=scoring_model
                        )
                        item["query"] = {
                            "query": question,
                            "tools": ai_tools,
                            "mode": mode,
                            "max_execution_time": get_mode_serving_budget(mode),
                            "date_filter_type": ai_date_filter.value,
                        }
                    else:  # web_search
                        question = await self.questions_dataset.generate_new_question_with_openai(
                            ["Web Search"], model=scoring_model
                        )
                        item["query"] = {"query": question}
                except Exception as e:
                    bt.logging.error(
                        f"[SyntheticGen] Failed to generate "
                        f"{item['search_type']} question: {e}"
                    )

        if llm_items:
            bt.logging.info(
                f"[SyntheticGen] Generating {len(llm_items)} LLM questions "
                f"(concurrency={self.MAX_CONCURRENT_LLM})..."
            )
            await asyncio.gather(*[_generate_one(item) for item in llm_items])

        # Drop items where generation failed (query stayed None)
        items = [i for i in items if i["query"] is not None]

        bt.logging.info(
            f"[SyntheticGen] Generated {len(items)} queries "
            f"({len(items) - len(llm_items)} instant, {len(llm_items)} LLM)"
        )
        return items

    def _generate_dataset_queries(
        self,
        available_uids: List[int],
        verified_by_type: dict[str, dict[int, int]],
    ) -> List[dict] | None:
        def count(st: str) -> int:
            return sum(
                verified_by_type.get(st, {}).get(uid, 1) for uid in available_uids
            )

        # X dataset feeds AI-search's Twitter tool (advanced search), NOT basic x_search.
        x_rows = self.hf_pool.sample_lane(X_LANE, count("ai_search")) or []
        web_rows = self._sample_web(count("web_search"))
        ai_rows = self._sample_web(count("ai_search"))
        if not (x_rows or web_rows or ai_rows):
            return None

        web_cursor = ai_cursor = 0
        items: List[dict] = []

        for uid in available_uids:
            for search_type in SEARCH_TYPES:
                n = verified_by_type.get(search_type, {}).get(uid, 1)
                for _ in range(n):
                    if search_type == "x_search":
                        query = {"query": self.basic_dataset.generate_random_x_query()}
                    elif search_type == "web_search":
                        if not web_rows:
                            continue
                        row = web_rows[web_cursor % len(web_rows)]
                        web_cursor += 1
                        query = self._row_query(row)
                    else:
                        ai_tools = [random.choice([WEB_TOOL, TWITTER_TOOL])]
                        row = self._pick_ai_row(ai_tools, x_rows, ai_rows, ai_cursor)
                        if row is None:
                            continue
                        ai_cursor += 1
                        query = self._row_query(row)
                        query["tools"] = ai_tools

                    items.append(
                        {"uid": uid, "search_type": search_type, "query": query}
                    )

        if not items:
            return None

        bt.logging.info(
            f"[SyntheticGen] Generated {len(items)} dataset queries "
            f"(x={len(x_rows)}, web={len(web_rows)}, ai={len(ai_rows)} pool rows)"
        )
        return items

    def _sample_web(self, n: int) -> List[dict]:
        rows: List[dict] = []
        for lane in WEB_LANES:
            lane_rows = self.hf_pool.sample_lane(lane, n)
            if lane_rows:
                rows.extend(lane_rows)
        random.shuffle(rows)
        return rows[:n] if n else rows

    def _pick_ai_row(
        self,
        ai_tools: list[str],
        x_rows: List[dict],
        ai_rows: List[dict],
        cursor: int,
    ) -> dict | None:
        if TWITTER_TOOL in ai_tools and x_rows:
            return x_rows[cursor % len(x_rows)]
        if ai_rows:
            return ai_rows[cursor % len(ai_rows)]
        if x_rows:
            return x_rows[cursor % len(x_rows)]
        return None

    @staticmethod
    def _row_query(row: dict) -> dict:
        return {
            "query": row["question"],
            "start_date": row["start_date"],
            "end_date": row["end_date"],
        }
