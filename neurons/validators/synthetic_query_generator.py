import asyncio
import random
from typing import List

import bittensor as bt

from desearch.dataset import BasicQuestionsDataset, QuestionsDataset
from desearch.dataset.date_filters import random_date_filters

# Tool combinations for AI search scoring — weighted by frequency.
# Chosen once per epoch so every miner is evaluated on the same tools.
AI_SEARCH_TOOL_SETS = [
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

SEARCH_TYPES = ["ai_search", "x_search", "web_search"]


class SyntheticQueryGenerator:
    """
    Generates synthetic scoring queries locally using the existing
    desearch/dataset module + OpenAI for question enhancement.

    Replaces the centralized utility API for question generation.
    Each validator independently generates its own synthetics.

    Epoch-level parameters (tools, date_filter for ai_search) are chosen
    once and shared across all miners — only the question text differs.
    """

    MAX_CONCURRENT_LLM = 20  # Throttle concurrent OpenAI calls

    def __init__(self):
        self.questions_dataset = QuestionsDataset()
        self.basic_dataset = BasicQuestionsDataset()

    async def generate_epoch_queries(
        self,
        available_uids: List[int],
        spread_seconds: float = 55 * 60,
    ) -> List[dict]:
        """
        Batch-generate all synthetic queries for one scoring epoch.

        Epoch-level parameters (tools, date_filter for ai_search) are chosen
        once and shared across all miners — only the question text differs.
        Questions are generated concurrently (throttled by semaphore).

        Returns items sorted by fire-time delay, each containing:
            uid, search_type, query (dict), delay_seconds, scoring_seed
        """
        # --- Epoch-level params for ai_search (same for every miner) ---
        ai_tools = random.choice(AI_SEARCH_TOOL_SETS)
        ai_date_filter = random.choice(random_date_filters)

        bt.logging.info(
            f"[SyntheticGen] Epoch params: ai_tools={ai_tools} "
            f"date_filter={ai_date_filter.value}"
        )

        # --- Build items with random fire times ---
        items: List[dict] = []
        llm_items: List[dict] = []  # Only ai_search + web_search need LLM

        for uid in available_uids:
            for search_type in SEARCH_TYPES:
                item = {
                    "uid": uid,
                    "search_type": search_type,
                    "delay_seconds": random.uniform(0, spread_seconds),
                    "scoring_seed": random.randint(0, 2**31 - 1),
                    "query": None,
                }

                if search_type == "x_search":
                    # Instant — no LLM needed
                    item["query"] = {
                        "query": self.basic_dataset.generate_random_x_query()
                    }
                else:
                    llm_items.append(item)

                items.append(item)

        # --- Batch LLM generation with throttling ---
        semaphore = asyncio.Semaphore(self.MAX_CONCURRENT_LLM)

        async def _generate_one(item: dict) -> None:
            async with semaphore:
                try:
                    if item["search_type"] == "ai_search":
                        question = await self.questions_dataset.generate_new_question_with_openai(
                            ai_tools
                        )
                        item["query"] = {
                            "query": question,
                            "tools": ai_tools,
                            "date_filter_type": ai_date_filter.value,
                        }
                    else:  # web_search
                        question = await self.questions_dataset.generate_new_question_with_openai(
                            ["Web Search"]
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

        items.sort(key=lambda x: x["delay_seconds"])

        bt.logging.info(
            f"[SyntheticGen] Generated {len(items)} queries "
            f"({len(items) - len(llm_items)} instant, {len(llm_items)} LLM)"
        )
        return items
