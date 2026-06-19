import asyncio
import random
from typing import List

import bittensor as bt

from desearch.dataset import BasicQuestionsDataset, QuestionsDataset
from desearch.dataset.date_filters import random_date_filters
from desearch.protocol import ScoringModel

AI_SEARCH_TOOL_SETS = [
    ["Twitter Search"],
    ["Twitter Search", "Web Search"],
    ["Web Search"],
]

SEARCH_TYPES = ["ai_search", "x_search", "web_search"]


class SyntheticQueryGenerator:
    """
    Generates synthetic scoring queries locally using the existing
    desearch/dataset module + OpenAI for question enhancement.
    """

    MAX_CONCURRENT_LLM = 20  # Throttle concurrent OpenAI calls

    def __init__(self):
        self.questions_dataset = QuestionsDataset()
        self.basic_dataset = BasicQuestionsDataset()

    async def generate_epoch_queries(
        self,
        available_uids: List[int],
        verified_by_type: dict[str, dict[int, int]] | None = None,
        scoring_model: ScoringModel = ScoringModel.OPENAI_GPT4_1_NANO,
    ) -> List[dict]:
        if verified_by_type is None:
            verified_by_type = {}

        ai_tools = random.choice(AI_SEARCH_TOOL_SETS)
        ai_date_filter = random.choice(random_date_filters)

        bt.logging.info(
            f"[SyntheticGen] Epoch params: ai_tools={ai_tools} "
            f"date_filter={ai_date_filter.value}"
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
                        question = await self.questions_dataset.generate_new_question_with_openai(
                            ai_tools, model=scoring_model
                        )
                        item["query"] = {
                            "query": question,
                            "tools": ai_tools,
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
