"""Reword pool questions so the published wording is not the asked wording."""

import asyncio

import bittensor as bt

from desearch.protocol import ScoringModel
from desearch.utils import call_scoring_llm

MAX_CONCURRENT = 20
MAX_LENGTH_RATIO = 2.0
MIN_LENGTH_RATIO = 0.5

REWRITE_PROMPT = """Rewrite the search question so it asks for exactly the same thing in different words.

Rules:
- Keep every named entity, number, date and qualifier exactly as written.
- Do not add, drop or loosen any condition.
- Change the sentence structure and word choice, not just capitalisation or punctuation.
- Keep it about the same length and as plainly worded as the original.
- Do not answer the question, explain, or add commentary.

Reply with the rewritten question and nothing else."""


def _is_usable(original: str, rewritten: str) -> bool:
    if not rewritten:
        return False
    ratio = len(rewritten) / max(len(original), 1)
    return MIN_LENGTH_RATIO <= ratio <= MAX_LENGTH_RATIO


def _clean(text: str) -> str:
    return (text or "").strip().strip('"').strip()


async def rewrite_question(question: str, model: ScoringModel) -> str:
    """Reword a question, falling back to the original."""
    try:
        response = await call_scoring_llm(
            [
                {"role": "system", "content": REWRITE_PROMPT},
                {"role": "user", "content": question},
            ],
            model=model,
            temperature=1.0,
        )
    except Exception as e:
        bt.logging.warning(f"[Rewriter] Rewrite failed: {e}")
        return question

    rewritten = _clean(response)

    return rewritten if _is_usable(question, rewritten) else question


async def rewrite_questions(
    questions: list[str], model: ScoringModel
) -> dict[str, str]:
    """Map each distinct question to its rewrite."""
    distinct = list(dict.fromkeys(q for q in questions if q))
    if not distinct:
        return {}

    semaphore = asyncio.Semaphore(MAX_CONCURRENT)

    async def one(question: str) -> str:
        async with semaphore:
            return await rewrite_question(question, model)

    mapping = dict(zip(distinct, await asyncio.gather(*[one(q) for q in distinct])))

    changed = sum(1 for q, r in mapping.items() if q != r)
    bt.logging.info(f"[Rewriter] Rewrote {changed}/{len(distinct)} questions")

    return mapping
