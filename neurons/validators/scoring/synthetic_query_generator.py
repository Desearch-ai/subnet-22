import asyncio
import random
from collections import Counter
from typing import List

import bittensor as bt

from desearch.dataset import BasicQuestionsDataset, QuestionsDataset
from desearch.miner_config import AI_MODES, LANES, SearchType, lane_key
from desearch.dataset.date_filters import random_date_filters
from desearch.dataset.hf_dataset import HFQuestionPool
from desearch.protocol import ResultType, ScoringModel
from desearch.utils import (
    SearchMode,
    get_mode_serving_budget,
)

WEB_TOOL = "Web Search"
TWITTER_TOOL = "Twitter Search"

SEARCH_TYPES = ["ai_search", "x_search"]

X_LANE = "x"
WEB_LANES = ("news", "squad", "nq")


random_result_types = list(
    Counter(
        {
            ResultType.LINKS_WITH_FINAL_SUMMARY: 4,
            ResultType.ONLY_LINKS: 1,
        }
    ).elements()
)


_MODE_TOOL_WEIGHTS = {
    SearchMode.FAST: {WEB_TOOL: 1.0},
    SearchMode.BALANCED: {WEB_TOOL: 0.50, TWITTER_TOOL: 0.50},
    SearchMode.DEEP: {WEB_TOOL: 0.50, TWITTER_TOOL: 0.50},
}


def tools_for_mode(mode: SearchMode) -> list[str]:
    return list(_MODE_TOOL_WEIGHTS[mode])


def pick_ai_mode_and_tool() -> tuple[SearchMode, list[str]]:
    mode = random.choice(AI_MODES)
    return mode, [random.choice(tools_for_mode(mode))]


_AI_RESULT_WEIGHTS = {
    ResultType.LINKS_WITH_FINAL_SUMMARY: 0.80,
    ResultType.ONLY_LINKS: 0.20,
}


def _weighted_counts(n: int, weights: list[float]) -> list[int]:
    counts = [int(n * w) for w in weights]
    remainders = [n * w - c for w, c in zip(weights, counts)]
    idxs = list(range(len(weights)))
    for _ in range(n - sum(counts)):
        total = sum(remainders[i] for i in idxs)
        pick = random.random() * total
        acc = 0.0
        for i in idxs:
            acc += remainders[i]
            if pick <= acc:
                counts[i] += 1
                idxs.remove(i)
                break
    return counts


def _weighted_list(n: int, choices: dict) -> list:
    values = list(choices)
    out = []
    for value, count in zip(values, _weighted_counts(n, list(choices.values()))):
        out.extend([value] * count)
    random.shuffle(out)
    return out


def _ai_combos(mode: SearchMode, n: int) -> list[tuple[list[str], str]]:
    if n <= 0:
        return []
    tools = _weighted_list(n, _MODE_TOOL_WEIGHTS[mode])
    result_types = _weighted_list(n, _AI_RESULT_WEIGHTS)
    return [([tools[i]], result_types[i].value) for i in range(n)]


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
        llm_items: List[dict] = []  # Only ai_search needs LLM

        for uid in available_uids:
            for lane in LANES:
                search_type, mode = lane
                n = verified_by_type.get(lane_key(lane), {}).get(uid, 1)
                if search_type == SearchType.X_SEARCH:
                    for _ in range(n):
                        items.append(
                            {
                                "uid": uid,
                                "search_type": "x_search",
                                "query": {
                                    "query": self.basic_dataset.generate_random_x_query()
                                },
                            }
                        )
                    continue
                for ai_tools, result_type in _ai_combos(mode, n):
                    item = {
                        "uid": uid,
                        "search_type": "ai_search",
                        "query": None,
                        "_combo": (mode, ai_tools, result_type),
                    }
                    llm_items.append(item)
                    items.append(item)

        semaphore = asyncio.Semaphore(self.MAX_CONCURRENT_LLM)

        async def _generate_one(item: dict) -> None:
            async with semaphore:
                try:
                    mode, ai_tools, result_type = item["_combo"]
                    question = (
                        await self.questions_dataset.generate_new_question_with_openai(
                            ai_tools, model=scoring_model
                        )
                    )
                    item["query"] = {
                        "query": question,
                        "tools": ai_tools,
                        "mode": mode,
                        "max_execution_time": get_mode_serving_budget(mode),
                        "date_filter_type": ai_date_filter.value,
                        "result_type": result_type,
                    }
                except Exception as e:
                    bt.logging.error(
                        f"[SyntheticGen] Failed to generate "
                        f"{item['search_type']} question: {e}"
                    )
                finally:
                    item.pop("_combo", None)

        if llm_items:
            bt.logging.info(
                f"[SyntheticGen] Generating {len(llm_items)} LLM questions "
                f"(concurrency={self.MAX_CONCURRENT_LLM})..."
            )
            await asyncio.gather(*[_generate_one(item) for item in llm_items])

        # Drop items where generation failed (query stayed None)
        items = [i for i in items if i["query"] is not None]

        instant = sum(1 for i in items if i["search_type"] == "x_search")
        bt.logging.info(
            f"[SyntheticGen] Generated {len(items)} queries "
            f"({instant} instant, {len(items) - instant} LLM)"
        )
        return items

    def _generate_dataset_queries(
        self,
        available_uids: List[int],
        verified_by_type: dict[str, dict[int, int]],
    ) -> List[dict] | None:
        def ai_count() -> int:
            return sum(
                verified_by_type.get(lane_key((SearchType.AI_SEARCH, mode)), {}).get(
                    uid, 1
                )
                for uid in available_uids
                for mode in AI_MODES
            )

        # X dataset feeds AI-search's Twitter tool (advanced search), NOT basic x_search.
        total_ai = ai_count()
        x_rows = self.hf_pool.sample_lane(X_LANE, total_ai) or []
        ai_rows = self._sample_web(total_ai)
        if not (x_rows or ai_rows):
            return None

        ai_cursor = 0
        items: List[dict] = []

        for uid in available_uids:
            for lane in LANES:
                search_type, mode = lane
                n = verified_by_type.get(lane_key(lane), {}).get(uid, 1)
                if search_type == SearchType.X_SEARCH:
                    for _ in range(n):
                        items.append(
                            {
                                "uid": uid,
                                "search_type": "x_search",
                                "query": {
                                    "query": self.basic_dataset.generate_random_x_query()
                                },
                            }
                        )
                    continue
                for ai_tools, result_type in _ai_combos(mode, n):
                    row = self._pick_ai_row(ai_tools, x_rows, ai_rows, ai_cursor)
                    if row is None:
                        continue
                    ai_cursor += 1
                    query = self._row_query(row)
                    query["tools"] = ai_tools
                    query["mode"] = mode
                    query["max_execution_time"] = get_mode_serving_budget(mode)
                    query["result_type"] = result_type
                    items.append(
                        {"uid": uid, "search_type": "ai_search", "query": query}
                    )

        if not items:
            return None

        bt.logging.info(
            f"[SyntheticGen] Generated {len(items)} dataset queries "
            f"(x={len(x_rows)}, ai={len(ai_rows)} pool rows)"
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
