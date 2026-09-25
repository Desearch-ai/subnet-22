import asyncio
import hashlib

import aiohttp
import numpy as np
import pytest
from aiohttp import web

from desearch.embedding import (
    INPUT_SCHEMA,
    OUTPUT_SCHEMA,
    EmbeddingModel,
    encode_vector,
    write_parquet,
)
from neurons.validators.embed import (
    EMBED_SAMPLES,
    EmbedValidator,
    check_rows,
    pick_samples,
)
from tests.local_http import serving

DIMS = 8
MODEL = EmbeddingModel("tiny", "tiny/tiny", "rev", DIMS, "tiny")
INPUTS = [
    {
        "text_id": f"p#chunk{i}",
        "page_key": "p",
        "url": "https://a.example/p",
        "content_sha1": "c" * 40,
        "kind": "chunk",
        "index": i,
        "text": f"passage {i}",
    }
    for i in range(30)
]


def truth(text: str) -> np.ndarray:
    seed = int(hashlib.sha256(text.encode()).hexdigest()[:8], 16)
    vector = np.random.default_rng(seed).normal(size=DIMS)
    return vector / np.linalg.norm(vector)


def upload(rows=INPUTS, vector=lambda row: encode_vector(truth(row["text"]))) -> bytes:
    return write_parquet(
        [{"text_id": row["text_id"], "vector": vector(row)} for row in rows],
        OUTPUT_SCHEMA,
    )


def test_every_vector_present_well_formed_and_unit_length_passes_the_row_check():
    counts, vectors = check_rows(INPUTS, upload(), DIMS)

    assert counts == {"returned": 30, "missing": 0, "duplicates": 0, "malformed": 0}
    assert len(vectors) == 30


@pytest.mark.parametrize(
    "body, reason",
    [
        (upload(INPUTS[:-1]), "vectors_missing"),
        (upload(INPUTS + INPUTS[:1]), "vectors_malformed"),
        (upload(vector=lambda row: b"\x00" * 6), "vectors_malformed"),
        (upload(vector=lambda row: encode_vector(np.ones(DIMS))), None),
        (
            upload(vector=lambda row: (np.ones(DIMS) * 3).astype("<f2").tobytes()),
            "vectors_malformed",
        ),
        (b"not parquet", "unreadable"),
        (None, "unreadable"),
    ],
)
def test_a_broken_upload_fails_before_any_sample_is_recomputed(body, reason):
    counts, _ = check_rows(INPUTS, body, DIMS)

    assert counts.get("reason") == reason


def test_the_sample_is_fixed_by_its_seed_and_capped():
    first = pick_samples(INPUTS, "seed")

    assert first == pick_samples(list(reversed(INPUTS)), "seed")
    assert first != pick_samples(INPUTS, "other seed")
    assert len(first) == EMBED_SAMPLES and len(pick_samples(INPUTS[:3], "seed")) == 3


class Reference:
    model = MODEL

    def __init__(self, fail: bool = False):
        self.fail = fail
        self.asked: list[str] = []

    async def embed(self, texts):
        if self.fail:
            raise RuntimeError("provider is down")
        self.asked += texts
        return np.array([truth(t) * 5 for t in texts])


class Api:
    hotkey = "5Validator"

    def __init__(self, job):
        self.job = job
        self.posts: list[tuple[str, dict]] = []

    async def post(self, path, body=None):
        self.posts.append((path, body))
        if path == "/v1/validation/lease":
            job, self.job = self.job, None
            return {"job": job}
        return {}


async def judged(body: bytes, reference: Reference, given: bytes | None = None):
    given = given or write_parquet(INPUTS, INPUT_SCHEMA)

    async def handler(request: web.Request) -> web.Response:
        return web.Response(body=given if request.path == "/input" else body)

    async with serving(handler) as base, aiohttp.ClientSession() as http:
        job = {
            "task_id": "t1",
            "miner": "5Miner",
            "model": "tiny",
            "download_url": base + "upload",
            "input": {
                "url": base + "input",
                "sha256": hashlib.sha256(
                    write_parquet(INPUTS, INPUT_SCHEMA)
                ).hexdigest(),
            },
        }
        api = Api(job)
        checker = EmbedValidator(api, http, {"tiny": reference})
        await checker.check_next_task()
        return api.posts[1:]


def test_honest_vectors_pass_on_a_recomputed_sample():
    reference = Reference()

    ((path, result),) = asyncio.run(judged(upload(), reference))

    assert path == "/v1/validation/t1/score"
    assert (result["verdict"], result["matched"]) == ("pass", EMBED_SAMPLES)
    assert result["min_similarity"] == pytest.approx(1.0, abs=1e-3)
    assert len(reference.asked) == EMBED_SAMPLES


def test_vectors_from_another_model_fail_as_a_mismatch():
    other = upload(vector=lambda row: encode_vector(truth(row["text"] + " other")))

    ((_, result),) = asyncio.run(judged(other, Reference()))

    assert (result["verdict"], result["reason"]) == ("fail", "vectors_mismatch")
    assert result["mismatched"] == EMBED_SAMPLES


def test_a_reference_outage_hands_the_job_back_instead_of_judging():
    ((path, body),) = asyncio.run(judged(upload(), Reference(fail=True)))

    assert (path, body) == ("/v1/validation/t1/release", {"reason": "provider"})


def test_an_input_that_does_not_match_its_hash_is_our_problem_not_the_miners():
    swapped = write_parquet(INPUTS[:1], INPUT_SCHEMA)

    ((path, body),) = asyncio.run(judged(upload(), Reference(), given=swapped))

    assert (path, body) == ("/v1/validation/t1/release", {"reason": "download"})


@pytest.mark.parametrize(
    "body",
    [
        upload(),
        upload(vector=lambda row: encode_vector(truth(row["text"] + " other"))),
        upload(INPUTS[:-1]),
        b"not parquet",
    ],
)
def test_the_task_api_accepts_every_result_the_checker_sends(body, monkeypatch):
    from pathlib import Path

    from neurons.validators.embed import compare

    monkeypatch.syspath_prepend(str(Path(__file__).parents[2] / "task-api"))
    from app.models import EmbedScore

    result, vectors = check_rows(INPUTS, body, DIMS)
    if "verdict" not in result:
        picked = pick_samples(INPUTS, "seed")
        result |= compare(picked, vectors, np.array([truth(r["text"]) for r in picked]))

    assert EmbedScore.model_validate(result).model_dump(exclude_unset=True) == result
