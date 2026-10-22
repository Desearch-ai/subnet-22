import asyncio
import base64
import hashlib

import aiohttp
import numpy as np
import pytest
from aiohttp import web

from desearch.embedding import (
    INPUT_SCHEMA,
    EmbeddingModel,
    encode_vector,
    write_parquet,
)
from desearch.manifest import OPEN_LIST_KEY
from neurons.validators import tasks
from neurons.validators.embed import (
    EMBED_SAMPLES,
    LONG_CHARS,
    LONG_SAMPLES,
    MATCH_SIMILARITY,
    RANGE_READS,
    EmbedValidator,
    pick_samples,
)
from neurons.validators.ledger import Ledger
from neurons.validators.tasks import UPLOAD_CHANGED, UploadChanged
from tests.local_http import serving

DIMS = 8
MODEL = EmbeddingModel("tiny", "tiny/tiny", "rev", DIMS, 16)
SLICE = MODEL.vector_bytes
INPUTS = [
    {
        "text_id": f"p{i}",
        "page_key": f"p{i}",
        "url": f"https://a.example/p{i}",
        "content_sha1": "c" * 40,
        "kind": "page",
        "index": 0,
        "text": f"page {i}",
    }
    for i in range(60)
]
GIVEN = write_parquet(INPUTS, INPUT_SCHEMA)
SAMPLED = {row["text_id"] for row in pick_samples(INPUTS, "seed")}
UNSAMPLED = next(row for row in INPUTS if row["text_id"] not in SAMPLED)


def truth(text: str) -> np.ndarray:
    seed = int(hashlib.sha256(text.encode()).hexdigest()[:8], 16)
    vector = np.random.default_rng(seed).normal(size=DIMS)
    return vector / np.linalg.norm(vector)


def upload(rows=INPUTS, vector=lambda row: encode_vector(truth(row["text"]))) -> bytes:
    return b"".join(vector(row) for row in rows)


def but(text_ids: set[str], raw: bytes):
    """Honest vectors except raw for the given texts."""
    return (
        lambda row: raw
        if row["text_id"] in text_ids
        else encode_vector(truth(row["text"]))
    )


NAN = np.full(DIMS, np.nan).astype("<f2").tobytes()
ETAG = '"0cc175b9c0f1b6a831c399e269772661"'
PRECONDITION_PAGE = (
    b"<!DOCTYPE html><html><body>412 Precondition Failed</body></html>" * 300
)


def test_the_sample_is_fixed_by_its_seed_and_capped():
    first = pick_samples(INPUTS, "seed")

    assert first == pick_samples(list(reversed(INPUTS)), "seed")
    assert first != pick_samples(INPUTS, "other seed")
    assert len(first) == EMBED_SAMPLES and len(pick_samples(INPUTS[:3], "seed")) == 3


def test_every_sample_includes_long_texts_when_the_task_has_them():
    long = [{**row, "text": "x" * LONG_CHARS + row["text"]} for row in INPUTS[:6]]
    inputs = long + INPUTS[6:]

    picked = pick_samples(inputs, "seed")
    ids = [row["text_id"] for row in picked]

    assert len(ids) == len(set(ids)) == EMBED_SAMPLES
    assert sum(len(row["text"]) >= LONG_CHARS for row in picked) >= LONG_SAMPLES
    assert picked == pick_samples(list(reversed(inputs)), "seed")


class Reference:
    model = MODEL

    def __init__(self, fail: bool = False):
        self.fail = fail
        self.asked: list[str] = []

    async def vectors(self, job, picked):
        if self.fail:
            raise RuntimeError("provider is down")
        self.asked += [row["text"] for row in picked]
        return np.array([truth(row["text"]) * 5 for row in picked])


class Api:
    hotkey = "5Validator"

    def __init__(self):
        self.posts: list[tuple[str, dict]] = []

    async def post(self, path, body=None):
        self.posts.append((path, body))
        return {}


def stored(
    body: bytes | None,
    ranges: list,
    live: dict | None = None,
    current: str = ETAG,
    pins: list | None = None,
    refused: int = 412,
    listed=(),
):
    """Storage holding the input and the upload; a rewritten upload is refused with `refused`: 412 while it stays readable, 403 once storage closes it."""

    async def handler(request: web.Request) -> web.Response:
        if request.path == f"/{OPEN_LIST_KEY}":
            return web.json_response({"uploads": [{"task_id": t} for t in listed]})
        pinned = request.headers.get("If-Match")
        if pins is not None:
            pins.append((request.path, pinned))
        if request.path == "/input":
            return web.Response(body=GIVEN)
        if body is None:
            return web.Response(status=404)
        if pinned is not None and pinned != current:
            return web.Response(
                status=refused, body=PRECONDITION_PAGE, content_type="text/html"
            )
        asked = request.headers.get("Range")
        ranges.append(asked)
        if live is not None:
            live["now"] += 1
            live["most"] = max(live["most"], live["now"])
            await asyncio.sleep(0.005)
            live["now"] -= 1
        if asked is None:
            return web.Response(body=body)
        start, end = (int(n) for n in asked.removeprefix("bytes=").split("-"))
        if start >= len(body):
            return web.Response(
                status=416, headers={"Content-Range": f"bytes */{len(body)}"}
            )
        end = min(end, len(body) - 1)
        return web.Response(
            status=206,
            body=body[start : end + 1],
            headers={"Content-Range": f"bytes {start}-{end}/{len(body)}"},
        )

    return handler


async def judged(
    body: bytes | None,
    reference: Reference,
    given: bytes = GIVEN,
    ledger: Ledger | None = None,
    live: dict | None = None,
    etag: str | None = ETAG,
    current: str = ETAG,
    pins: list | None = None,
    refused: int = 412,
    listed=(),
):
    ranges: list = []
    async with (
        serving(stored(body, ranges, live, current, pins, refused, listed)) as base,
        aiohttp.ClientSession() as http,
    ):
        job = {
            "task_id": "t1",
            "key": "uploads/dt=2026-09-25/task=t1/5Miner-1.f16",
            "kind": "embed",
            "miner": "5Miner",
            "model": "tiny",
            "seed": "seed",
            "texts": len(INPUTS),
            "chars": 1000,
            "completed_at": 100.0,
            "download_url": base + "upload",
            "input": {
                "url": base + "input",
                "sha256": hashlib.sha256(given).hexdigest(),
            },
        }
        if etag:
            job["etag"] = etag
        api = Api()
        checker = EmbedValidator(
            api, http, {"tiny": reference}, ledger=ledger, storage_url=base
        )
        await checker.check(job)
        return api.posts, checker, ranges


def scored(body: bytes | None, reference: Reference | None = None) -> dict:
    ((path, result),), _, _ = asyncio.run(judged(body, reference or Reference()))
    assert path == "/v1/validation/t1/score"
    return result


def test_honest_vectors_pass_and_only_the_sampled_slices_are_read():
    reference = Reference()

    ((_, result),), _, ranges = asyncio.run(judged(upload(), reference))

    assert (result["verdict"], result["reason"]) == ("pass", "ok")
    assert (result["sampled"], result["matched"]) == (EMBED_SAMPLES, EMBED_SAMPLES)
    assert result["min_similarity"] == pytest.approx(1.0, abs=1e-3)
    assert len(reference.asked) == EMBED_SAMPLES
    rows = {row["text_id"]: i for i, row in enumerate(INPUTS)}
    assert sorted(ranges) == sorted(
        f"bytes={rows[t] * SLICE}-{rows[t] * SLICE + SLICE - 1}" for t in SAMPLED
    ), "one range per sampled text, never the whole upload"
    assert all(
        sample["index"] == rows[sample["text_id"]] for sample in result["samples"]
    )


def test_the_result_holds_exactly_the_fields_the_task_api_takes():
    result = scored(upload(vector=but({next(iter(SAMPLED))}, NAN)))

    assert set(result) == {
        "verdict",
        "reason",
        "sampled",
        "matched",
        "mismatched",
        "malformed",
        "min_similarity",
        "samples",
    }
    assert all(
        set(sample) == {"text_id", "index", "outcome", "similarity"}
        for sample in result["samples"]
    )
    assert (
        result["sampled"]
        == len(result["samples"])
        == (result["matched"] + result["mismatched"] + result["malformed"])
    )


def test_vectors_from_another_model_fail_as_a_mismatch():
    result = scored(
        upload(vector=lambda row: encode_vector(truth(row["text"] + " other")))
    )

    assert (result["verdict"], result["reason"]) == ("fail", "vectors_mismatch")
    assert result["mismatched"] == EMBED_SAMPLES


@pytest.mark.parametrize(
    "body",
    [upload(INPUTS[:-1]), upload(INPUTS + INPUTS[:1]), upload() + b"\x00", b""],
    ids=["one short", "one extra", "a byte extra", "empty"],
)
def test_an_upload_of_the_wrong_size_fails_without_asking_the_reference(body):
    reference = Reference()

    result = scored(body, reference)

    assert (result["verdict"], result["reason"]) == ("fail", "wrong_size")
    assert (result["sampled"], result["samples"]) == (0, [])
    assert reference.asked == []


@pytest.mark.parametrize(
    "raw",
    [NAN, (np.ones(DIMS) * 3).astype("<f2").tobytes(), b"\x00" * SLICE],
    ids=["nan", "not length 1", "zeros"],
)
def test_a_sampled_vector_that_is_not_a_finite_unit_vector_is_malformed(raw):
    bad = next(iter(SAMPLED))
    reference = Reference()

    result = scored(upload(vector=but({bad}, raw)), reference)

    assert (result["verdict"], result["reason"]) == ("fail", "vectors_malformed")
    assert (result["malformed"], result["matched"]) == (1, EMBED_SAMPLES - 1)
    (sample,) = [s for s in result["samples"] if s["outcome"] == "malformed"]
    assert (sample["text_id"], sample["similarity"]) == (bad, None)
    assert len(reference.asked) == EMBED_SAMPLES - 1, (
        "only the well-formed are recomputed"
    )


def test_a_broken_vector_outside_the_sample_is_never_read():
    result = scored(upload(vector=but({UNSAMPLED["text_id"]}, NAN)))

    assert (result["verdict"], result["matched"]) == ("pass", EMBED_SAMPLES)


def test_a_reference_outage_leaves_the_upload_for_later_instead_of_judging():
    posts, checker, _ = asyncio.run(judged(upload(), Reference(fail=True)))

    assert posts == [] and "t1" in checker.deferred
    assert checker.provider_failures == 1


def test_an_input_that_does_not_match_its_hash_is_our_problem_not_the_miners():
    swapped = write_parquet(INPUTS[:1], INPUT_SCHEMA)

    posts, checker, ranges = asyncio.run(judged(upload(), Reference(), given=swapped))

    assert posts == [] and "t1" in checker.deferred and ranges == []


def test_a_lost_upload_is_handed_back_to_the_task_api():
    posts, checker, _ = asyncio.run(judged(None, Reference()))

    assert posts == [("/v1/validation/t1/release", {"reason": "missing"})]
    assert "t1" in checker.deferred


def test_the_upload_is_read_only_in_the_version_its_completion_pinned():
    pins: list = []

    ((_, result),), _, _ = asyncio.run(judged(upload(), Reference(), pins=pins))

    assert result["verdict"] == "pass"
    assert {pin for path, pin in pins if path == "/upload"} == {ETAG}
    assert [pin for path, pin in pins if path == "/input"] == [None], (
        "the input is the task API's file, not the miner's upload"
    )


def test_a_manifest_without_an_etag_reads_without_if_match():
    pins: list = []

    ((_, result),), _, _ = asyncio.run(
        judged(upload(), Reference(), etag=None, current='"rewritten"', pins=pins)
    )

    assert result["verdict"] == "pass"
    assert {pin for _, pin in pins} == {None}


@pytest.mark.parametrize(
    "refused", [412, 403], ids=["precondition failed", "closed when rewritten"]
)
def test_an_upload_changed_after_its_completion_fails_and_costs_its_characters(
    refused,
):
    reference = Reference()
    ledger = Ledger(":memory:")

    ((path, result),), _, ranges = asyncio.run(
        judged(
            upload(),
            reference,
            ledger=ledger,
            current='"rewritten"',
            refused=refused,
            listed=("t1",),
        )
    )

    assert path == "/v1/validation/t1/score"
    assert result == {
        "verdict": "fail",
        "reason": UPLOAD_CHANGED,
        "sampled": 0,
        "matched": 0,
        "mismatched": 0,
        "malformed": 0,
        "min_similarity": None,
        "samples": [],
    }
    assert ranges == [] and reference.asked == [], "no 412 body is read as a vector"
    check = ledger.db.execute(
        "SELECT kind, verdict, credited, assigned FROM checks"
    ).fetchone()
    assert check == ("embed", "fail", 0, 1000)
    assert ledger.embed_paid(200.0)["5Miner"].amount == -1000


def test_a_pinned_read_refused_after_its_task_closed_is_missing_not_changed():
    reference = Reference()
    ledger = Ledger(":memory:")

    ((path, body),), _, ranges = asyncio.run(
        judged(upload(), reference, ledger=ledger, current='"gone"', refused=403)
    )

    assert (path, body) == ("/v1/validation/t1/release", {"reason": "missing"})
    assert ranges == [] and reference.asked == []
    assert ledger.db.execute("SELECT COUNT(*) FROM checks").fetchone() == (0,)


def test_a_read_that_failed_beside_a_changed_one_does_not_hide_the_change(
    monkeypatch,
):
    monkeypatch.setattr(tasks, "RETRY_DELAY_S", 0)

    async def handler(request: web.Request) -> web.Response:
        if request.headers["Range"].startswith("bytes=0-"):
            return web.Response(status=503)
        return web.Response(status=412, body=PRECONDITION_PAGE)

    async def read():
        async with serving(handler) as base, aiohttp.ClientSession() as http:
            job = {"task_id": "t1", "download_url": base + "upload", "etag": ETAG}
            await EmbedValidator(Api(), http, {}).read_vectors(job, [0, 1, 2], SLICE)

    with pytest.raises(UploadChanged):
        asyncio.run(read())


def test_the_sampled_slices_are_read_a_few_at_a_time():
    live = {"now": 0, "most": 0}

    ((_, result),), _, _ = asyncio.run(judged(upload(), Reference(), live=live))

    assert result["verdict"] == "pass"
    assert 1 < live["most"] <= RANGE_READS


def test_a_vector_inflated_within_the_norm_tolerance_is_judged_on_its_cosine():
    def slanted(row) -> bytes:
        exact = truth(row["text"])
        other = truth(row["text"] + " other")
        other -= (other @ exact) * exact
        other /= np.linalg.norm(other)
        unit = 0.982 * exact + np.sqrt(1 - 0.982**2) * other
        return (unit * 1.0089).astype("<f2").tobytes()

    result = scored(upload(vector=slanted))

    assert (result["verdict"], result["reason"]) == ("fail", "vectors_mismatch")
    assert result["min_similarity"] < MATCH_SIMILARITY


def near(row, cosine: float) -> bytes:
    exact = truth(row["text"])
    other = truth(row["text"] + " other")
    other -= (other @ exact) * exact
    other /= np.linalg.norm(other)
    return encode_vector(cosine * exact + np.sqrt(1 - cosine**2) * other)


def test_vectors_a_little_off_the_model_fail_and_the_same_model_on_other_hardware_passes():
    failed = scored(upload(vector=lambda row: near(row, 0.998)))
    passed = scored(upload(vector=lambda row: near(row, 0.99995)))

    assert (failed["verdict"], failed["reason"]) == ("fail", "vectors_mismatch")
    assert (passed["verdict"], passed["matched"]) == ("pass", EMBED_SAMPLES)


def test_a_check_is_kept_in_characters_for_the_validators_own_weights():
    ledger = Ledger(":memory:")
    asyncio.run(judged(upload(), Reference(), ledger=ledger))

    check = ledger.db.execute(
        "SELECT kind, verdict, credited, returned, content, assigned FROM checks"
    ).fetchone()
    assert check == ("embed", "pass", 1000, 1000, 1000, 1000)

    asyncio.run(judged(upload(INPUTS[:-1]), Reference(), ledger=ledger))
    check = ledger.db.execute(
        "SELECT kind, verdict, credited, assigned FROM checks"
    ).fetchone()
    assert check == ("embed", "fail", 0, 1000)
    assert ledger.embed_paid(200.0)["5Miner"].amount == -1000


def test_a_validator_without_a_model_server_gets_its_sample_from_the_task_api():
    from neurons.validators.embed import ApiReference

    class Api:
        def __init__(self):
            self.posts = []

        async def post(self, path, body=None):
            self.posts.append((path, body))
            return {
                "vectors": [
                    {
                        "text_id": i,
                        "embedding": base64.b64encode(
                            truth(i).astype("<f4").tobytes()
                        ).decode(),
                    }
                    for i in reversed(body["text_ids"])
                ]
            }

    api = Api()
    picked = INPUTS[:3]
    vectors = asyncio.run(ApiReference(api, MODEL).vectors({"task_id": "t1"}, picked))

    assert api.posts == [("/v1/embed/t1/reference", {"text_ids": ["p0", "p1", "p2"]})]
    assert np.allclose(vectors, [truth(row["text_id"]) for row in picked], atol=1e-6), (
        "in the order asked"
    )


def test_the_task_api_reference_is_waited_for_while_another_validator_computes_it():
    from desearch.client import TaskApiError
    from neurons.validators.embed import ApiReference

    class Api:
        def __init__(self, answers):
            self.answers = answers

        async def post(self, path, body=None):
            answer = self.answers.pop(0)
            if isinstance(answer, Exception):
                raise answer
            return {
                "vectors": [
                    {
                        "text_id": i,
                        "embedding": base64.b64encode(
                            truth(i).astype("<f4").tobytes()
                        ).decode(),
                    }
                    for i in body["text_ids"]
                ]
            }

    busy = TaskApiError(503, "the reference for this upload is being computed", 0.01)
    picked = INPUTS[:2]
    api = Api([busy, busy, None])
    vectors = asyncio.run(ApiReference(api, MODEL).vectors({"task_id": "t1"}, picked))
    assert api.answers == [] and vectors.shape == (2, MODEL.dims)

    down = TaskApiError(503, "the reference model server is unavailable", 30)
    with pytest.raises(TaskApiError):
        asyncio.run(ApiReference(Api([down]), MODEL).vectors({"task_id": "t1"}, picked))


@pytest.mark.parametrize("status", [409, 422, 429])
def test_a_reference_refused_for_one_upload_defers_it_without_pausing_the_checks(
    status,
):
    from desearch.client import TaskApiError
    from neurons.validators.embed import ApiReference

    class Refusing:
        async def post(self, path, body=None):
            raise TaskApiError(status, "no")

    reference = ApiReference(Refusing(), MODEL)
    posts, checker, _ = asyncio.run(judged(upload(), reference))

    assert posts == [] and "t1" in checker.deferred
    assert checker.provider_failures == 0


def test_a_validator_checks_only_the_models_its_own_server_serves():
    from neurons.validators.validator import Validator

    async def references(served):
        async def handler(request: web.Request) -> web.Response:
            return web.json_response({"data": [{"id": name} for name in served]})

        made = Validator.__new__(Validator)
        made.stopping = asyncio.Event()
        async with serving(handler) as base:
            found = await made.embed_references(base + "v1/embeddings")
        for client in found.values():
            await client.aclose()
        return sorted(found)

    assert asyncio.run(references(["desearch-4b-v1"])) == ["desearch-4b-v1"]
    assert asyncio.run(references(["Qwen/Qwen3-Embedding-4B"])) == []
