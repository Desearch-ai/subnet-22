"""
Import questions from a HuggingFace dataset into the questions table.

Usage:
    python -m app.scripts.import_huggingface \
        --dataset "squad" \
        --split "train" \
        --column "question" \
        --search-types ai_search \
        --ai-tools wikipedia web \
        --limit 1000

This is a base script — adapt the --column flag (and optionally the code
below) to match the structure of whichever HuggingFace dataset you are
importing.
"""

import argparse
import asyncio
from uuid import uuid4

from datasets import load_dataset

from app.db.session import engine
from app.domains.dataset.enums import AISearchTool, SearchType
from app.domains.dataset.models.question import Base, Question


async def import_dataset(
    dataset_name: str,
    split: str,
    column: str,
    search_types: list[SearchType],
    ai_tools: list[AISearchTool] | None,
    limit: int | None,
    batch_size: int = 500,
):
    # ── Load from HuggingFace ────────────────────────────────────────
    print(f"Loading HuggingFace dataset: {dataset_name} (split={split})")
    ds = load_dataset(dataset_name, split=split)

    rows = ds[column]
    if limit:
        rows = rows[:limit]

    # Deduplicate
    unique_queries = list(dict.fromkeys(rows))
    print(f"Loaded {len(rows)} rows, {len(unique_queries)} unique questions")

    source = f"huggingface:{dataset_name}"
    search_types_values = [st.value for st in search_types]
    ai_tools_values = [t.value for t in ai_tools] if ai_tools else None

    # ── Insert in batches ────────────────────────────────────────────
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    from app.db.session import async_session

    inserted = 0
    for i in range(0, len(unique_queries), batch_size):
        batch = unique_queries[i : i + batch_size]
        async with async_session() as session:
            async with session.begin():
                for query_text in batch:
                    if not query_text or not query_text.strip():
                        continue
                    q = Question(
                        id=uuid4(),
                        query=query_text.strip(),
                        search_types=search_types_values,
                        ai_search_tools=ai_tools_values,
                        source=source,
                    )
                    session.add(q)
                inserted += len(batch)
        print(f"  Inserted {inserted}/{len(unique_queries)}")

    print(f"Done. Total inserted: {inserted}")
    await engine.dispose()


def main():
    parser = argparse.ArgumentParser(
        description="Import HuggingFace dataset into questions table"
    )
    parser.add_argument(
        "--dataset", required=True, help="HuggingFace dataset name (e.g. 'squad')"
    )
    parser.add_argument(
        "--split", default="train", help="Dataset split (default: train)"
    )
    parser.add_argument(
        "--column", default="question", help="Column containing question text"
    )
    parser.add_argument(
        "--search-types",
        nargs="+",
        choices=[s.value for s in SearchType],
        required=True,
        help="Search types for these questions",
    )
    parser.add_argument(
        "--ai-tools",
        nargs="*",
        choices=[t.value for t in AISearchTool],
        default=None,
        help="AI search tools (only relevant for ai_search type)",
    )
    parser.add_argument("--limit", type=int, default=None, help="Max rows to import")
    args = parser.parse_args()

    asyncio.run(
        import_dataset(
            dataset_name=args.dataset,
            split=args.split,
            column=args.column,
            search_types=[SearchType(s) for s in args.search_types],
            ai_tools=[AISearchTool(t) for t in args.ai_tools]
            if args.ai_tools
            else None,
            limit=args.limit,
        )
    )


if __name__ == "__main__":
    main()
