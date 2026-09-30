from unittest.mock import AsyncMock, patch

import pytest

from desearch.protocol import ScoringModel
from neurons.validators.scoring.question_rewriter import (
    rewrite_question,
    rewrite_questions,
)
from neurons.validators.scoring.synthetic_query_generator import SyntheticQueryGenerator

MODEL = ScoringModel.OPENAI_GPT4_1_NANO
QUESTION = "Which Indian regulatory body issued show-cause notices in August 2026?"


def _llm(*responses):
    return patch(
        "neurons.validators.scoring.question_rewriter.call_scoring_llm",
        AsyncMock(side_effect=list(responses)),
    )


@pytest.mark.asyncio
async def test_rewrite_replaces_the_question():
    reworded = "In August 2026, which regulatory body in India sent show-cause notices?"

    with _llm(f'"{reworded}"  '):
        assert await rewrite_question(QUESTION, MODEL) == reworded


@pytest.mark.asyncio
@pytest.mark.parametrize("response", ["", "   ", "why?", "x" * 500, None])
async def test_unusable_rewrite_keeps_the_original(response):
    with _llm(response):
        assert await rewrite_question(QUESTION, MODEL) == QUESTION


@pytest.mark.asyncio
async def test_llm_failure_keeps_the_original():
    with _llm(RuntimeError("boom")):
        assert await rewrite_question(QUESTION, MODEL) == QUESTION


@pytest.mark.asyncio
async def test_each_distinct_question_is_rewritten_once():
    with _llm(
        "Which body in India sent show-cause notices during August 2026?",
        "About what did the second question actually inquire?",
    ) as llm:
        mapping = await rewrite_questions(
            [QUESTION, QUESTION, "What did the second question actually ask about?"],
            MODEL,
        )

    assert llm.await_count == 2
    assert (
        mapping[QUESTION]
        == "Which body in India sent show-cause notices during August 2026?"
    )
    assert (
        mapping["What did the second question actually ask about?"]
        == "About what did the second question actually inquire?"
    )


@pytest.mark.asyncio
async def test_same_row_asked_to_many_uids_gets_one_rewrite():
    items = [
        {"uid": 1, "search_type": "ai_search", "query": {"query": QUESTION}},
        {"uid": 2, "search_type": "ai_search", "query": {"query": QUESTION}},
        {"uid": 3, "search_type": "x_search", "query": {"query": "x lane question"}},
    ]

    with _llm("a differently worded question about notices"):
        await SyntheticQueryGenerator._rewrite_dataset_questions(items, MODEL)

    assert {i["query"]["query"] for i in items if i["search_type"] == "ai_search"} == {
        "a differently worded question about notices"
    }
    assert items[2]["query"]["query"] == "x lane question"
