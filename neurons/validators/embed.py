from __future__ import annotations

import asyncio
import base64
import hashlib
import logging
import random
from collections import Counter

import numpy as np

from desearch.client import TaskApiError
from desearch.embedding import (
    INPUT_SCHEMA,
    EmbeddingClient,
    EmbeddingModel,
    decode_vector,
    read_parquet,
)
from desearch.kinds import EMBED
from neurons.validators.tasks import (
    UPLOAD_CHANGED,
    DownloadFailed,
    TaskChecker,
    UploadChanged,
    UploadMissing,
)

log = logging.getLogger("validator")

EMBED_SAMPLES = 48
LONG_SAMPLES = 6
# Long enough to reach the model's token limit.
LONG_CHARS = 6000
MATCH_SIMILARITY = 0.999
NORM_TOLERANCE = 0.01
RANGE_READS = 16
# Another validator's request for the same upload is being embedded; it takes seconds.
BUSY_WAIT_S = 60.0
BUSY_RETRY_MAX_S = 5.0


class ReferenceRefused(Exception):
    """The task API gives no reference vectors for this upload; the model server is not at fault."""


def unit_vector(raw: bytes, dims: int) -> np.ndarray | None:
    """The vector in one slice of an upload, or None when it is not finite and of length 1."""
    vector = decode_vector(raw, dims)
    if (
        vector is None
        or not np.all(np.isfinite(vector))
        or abs(float(np.linalg.norm(vector)) - 1.0) > NORM_TOLERANCE
    ):
        return None
    return vector


def pick_samples(
    inputs: list[dict], seed: str, count: int = EMBED_SAMPLES
) -> list[dict]:
    """The same texts for every validator of the upload, a few of them long ones when it has any."""
    ordered = sorted(inputs, key=lambda row: row["text_id"])
    draw = random.Random(seed)
    long = [row for row in ordered if len(row["text"]) >= LONG_CHARS]
    picked = draw.sample(long, min(LONG_SAMPLES, len(long), count))
    taken = {row["text_id"] for row in picked}
    rest = [row for row in ordered if row["text_id"] not in taken]
    return picked + draw.sample(rest, min(count - len(picked), len(rest)))


def compare(
    picked: list[dict],
    indexes: list[int],
    vectors: list[np.ndarray | None],
    reference: np.ndarray,
) -> dict:
    """Each sampled vector against the reference vectors of the well-formed ones, in order."""
    expected = iter(reference)
    samples = []
    for row, index, vector in zip(picked, indexes, vectors, strict=True):
        sample = {"text_id": row["text_id"], "index": index}
        if vector is None:
            samples.append({**sample, "outcome": "malformed", "similarity": None})
            continue
        want = next(expected)
        want = want / max(float(np.linalg.norm(want)), 1e-12)
        mine = vector / max(float(np.linalg.norm(vector)), 1e-12)
        similarity = float(np.dot(mine, want))
        outcome = "matched" if similarity >= MATCH_SIMILARITY else "mismatched"
        samples.append(
            {**sample, "outcome": outcome, "similarity": round(similarity, 6)}
        )
    counts = Counter(sample["outcome"] for sample in samples)
    if counts["malformed"]:
        reason = "vectors_malformed"
    elif counts["mismatched"]:
        reason = "vectors_mismatch"
    else:
        reason = "ok"
    similarities = [s["similarity"] for s in samples if s["similarity"] is not None]
    return {
        "verdict": "pass" if reason == "ok" else "fail",
        "reason": reason,
        "sampled": len(samples),
        "matched": counts["matched"],
        "mismatched": counts["mismatched"],
        "malformed": counts["malformed"],
        "min_similarity": min(similarities, default=None),
        "samples": samples,
    }


def unsampled_fail(reason: str) -> dict:
    """A fail decided before any vector was compared."""
    return {
        **compare([], [], [], np.empty(0)),
        "verdict": "fail",
        "reason": reason,
    }


class ServerReference:
    """The validator's own model server."""

    def __init__(self, client: EmbeddingClient):
        self.client = client
        self.model = client.model

    async def vectors(self, job: dict, picked: list[dict]) -> np.ndarray:
        return await self.client.embed([row["text"] for row in picked])

    async def aclose(self) -> None:
        await self.client.aclose()


class ApiReference:
    """Desearch's model server through the task API, for a validator without one."""

    def __init__(self, api, model: EmbeddingModel):
        self.api = api
        self.model = model

    async def vectors(self, job: dict, picked: list[dict]) -> np.ndarray:
        ids = [row["text_id"] for row in picked]
        answer = await self.ask(f"/v1/embed/{job['task_id']}/reference", ids)
        given = {
            row["text_id"]: np.frombuffer(
                base64.b64decode(row["embedding"]), dtype="<f4"
            )
            for row in answer["vectors"]
        }
        vectors = [given[text_id] for text_id in ids]
        if any(vector.size != self.model.dims for vector in vectors):
            raise ValueError(f"reference vectors are not {self.model.dims} long")
        return np.array(vectors, dtype=np.float32)

    async def ask(self, path: str, ids: list[str]) -> dict:
        """The API's answer, waiting while another validator's request for the upload is embedded."""
        waited = 0.0
        while True:
            try:
                return await self.api.post(path, {"text_ids": ids})
            except TaskApiError as exc:
                if exc.status in (409, 422, 429):
                    raise ReferenceRefused(str(exc)) from exc
                pause = exc.retry_after
                if exc.status != 503 or pause is None or pause > BUSY_RETRY_MAX_S:
                    raise
                if waited >= BUSY_WAIT_S:
                    raise ReferenceRefused("its reference stayed busy") from exc
                await asyncio.sleep(pause)
                waited += pause

    async def aclose(self) -> None:
        pass


class EmbedValidator(TaskChecker):
    """Recomputes a sample of each embed task's vectors, on the validator's own model server or Desearch's."""

    kinds = (EMBED,)

    def __init__(
        self,
        api,
        http,
        references: dict[str, ServerReference | ApiReference],
        ledger=None,
        storage_url: str = "",
        seeds=None,
        signer: str = "",
    ):
        super().__init__(
            api,
            http,
            ledger=ledger,
            storage_url=storage_url,
            seeds=seeds,
            signer=signer,
        )
        self.references = references

    async def check(self, job: dict) -> dict | None:
        task_id = job["task_id"]
        reference = self.references.get(job["model"])
        if reference is None:
            log.warning("task=%s needs %s, which we cannot run", task_id, job["model"])
            self.defer(task_id, 3600)
            return {}

        try:
            given = await self.download(job["input"]["url"], task_id)
        except UploadMissing:
            await self.hand_back(task_id, "missing")
            return {}
        except DownloadFailed as exc:
            log.warning("%s, trying again later", exc)
            self.defer(task_id)
            return {}
        if given is None or hashlib.sha256(given).hexdigest() != job["input"]["sha256"]:
            log.warning("task=%s input does not match its hash", task_id)
            self.defer(task_id)
            return {}

        inputs = read_parquet(given, INPUT_SCHEMA)
        picked = pick_samples(inputs, job["seed"])
        index_of = {row["text_id"]: index for index, row in enumerate(inputs)}
        indexes = [index_of[row["text_id"]] for row in picked]
        size = reference.model.vector_bytes
        try:
            slices = await self.read_vectors(job, indexes, size)
        except UploadMissing:
            await self.hand_back(task_id, "missing")
            return {}
        except UploadChanged:
            log.warning("task=%s upload changed after it was completed", task_id)
            slices = None
        except DownloadFailed as exc:
            log.warning("%s, trying again later", exc)
            self.defer(task_id)
            return {}

        if slices is None:
            result = unsampled_fail(UPLOAD_CHANGED)
        elif any(total != len(inputs) * size for _, total in slices):
            result = unsampled_fail("wrong_size")
        else:
            vectors = [unit_vector(raw, reference.model.dims) for raw, _ in slices]
            formed = [row for row, v in zip(picked, vectors) if v is not None]
            try:
                expected = (
                    await reference.vectors(job, formed) if formed else np.empty(0)
                )
            except ReferenceRefused as exc:
                log.warning("task=%s no reference vectors: %s", task_id, exc)
                self.defer(task_id)
                return {}
            except Exception as exc:
                log.warning("task=%s reference failed: %s", task_id, exc)
                self.provider_failed()
                self.defer(task_id)
                return {}
            self.provider_worked()
            result = compare(picked, indexes, vectors, expected)

        await self.submit_verdict(job, result)
        self.note_verdict(job, result)
        log.info(
            "task=%s miner=%s embed %d texts, matched %d/%d (min %s), %d malformed,"
            " verdict=%s reason=%s",
            task_id,
            job["miner"][:10],
            len(inputs),
            result["matched"],
            result["sampled"],
            result["min_similarity"],
            result["malformed"],
            result["verdict"],
            result["reason"],
        )
        return result

    async def read_vectors(
        self, job: dict, indexes: list[int], size: int
    ) -> list[tuple[bytes, int]]:
        """Each sampled row's slice of the upload and the upload's total size, a few reads at a time."""
        slots = asyncio.Semaphore(RANGE_READS)

        async def read(index: int) -> tuple[bytes, int]:
            async with slots:
                return await self.read_range(
                    job["download_url"],
                    index * size,
                    size,
                    job["task_id"],
                    job.get("etag"),
                )

        found = await asyncio.gather(
            *(read(index) for index in indexes), return_exceptions=True
        )
        failed = [item for item in found if isinstance(item, BaseException)]
        if failed:
            # Proof of a rewritten upload outranks a read that merely failed.
            changed = [error for error in failed if isinstance(error, UploadChanged)]
            raise (changed or failed)[0]
        return found
