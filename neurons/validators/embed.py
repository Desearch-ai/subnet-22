from __future__ import annotations

import hashlib
import logging
import random

import numpy as np

from desearch.embedding import (
    INPUT_SCHEMA,
    OUTPUT_SCHEMA,
    EmbeddingClient,
    decode_vector,
    read_parquet,
)
from neurons.validators.tasks import DownloadFailed, TaskChecker, UploadMissing

log = logging.getLogger("validator")

EMBED_SAMPLES = 20
# Two independent hosts of the same weights agree above 0.9998; another model lands near 0.
MATCH_SIMILARITY = 0.99
NORM_TOLERANCE = 0.01


def check_rows(
    inputs: list[dict], upload: bytes | None, dims: int
) -> tuple[dict, dict[str, np.ndarray]]:
    """Counts for every row of the upload, and the usable vectors by text id."""
    counts = {"returned": 0, "missing": 0, "duplicates": 0, "malformed": 0}
    if upload is None:
        return {**counts, "verdict": "fail", "reason": "unreadable"}, {}
    try:
        rows = read_parquet(upload, OUTPUT_SCHEMA)
    except ValueError:
        return {**counts, "verdict": "fail", "reason": "unreadable"}, {}

    wanted = {row["text_id"] for row in inputs}
    vectors: dict[str, np.ndarray] = {}
    seen: set[str] = set()
    for row in rows:
        text_id = row["text_id"]
        if text_id in seen or text_id not in wanted:
            counts["duplicates"] += 1
            continue
        seen.add(text_id)
        vector = decode_vector(row["vector"], dims)
        if (
            vector is None
            or not np.all(np.isfinite(vector))
            or abs(float(np.linalg.norm(vector)) - 1.0) > NORM_TOLERANCE
        ):
            counts["malformed"] += 1
            continue
        vectors[text_id] = vector
    counts["returned"] = len(seen)
    counts["missing"] = len(wanted - seen)
    if counts["duplicates"] or counts["malformed"]:
        return {**counts, "verdict": "fail", "reason": "vectors_malformed"}, vectors
    if counts["missing"]:
        return {**counts, "verdict": "fail", "reason": "vectors_missing"}, vectors
    return counts, vectors


def pick_samples(
    inputs: list[dict], seed: str, count: int = EMBED_SAMPLES
) -> list[dict]:
    ordered = sorted(inputs, key=lambda row: row["text_id"])
    return random.Random(seed).sample(ordered, min(count, len(ordered)))


def compare(
    picked: list[dict], vectors: dict[str, np.ndarray], reference: np.ndarray
) -> dict:
    samples = []
    for row, expected in zip(picked, reference, strict=True):
        expected = expected / max(float(np.linalg.norm(expected)), 1e-12)
        mine = vectors[row["text_id"]]
        mine = mine / max(float(np.linalg.norm(mine)), 1e-12)
        similarity = float(np.dot(mine, expected))
        outcome = "matched" if similarity >= MATCH_SIMILARITY else "mismatched"
        samples.append(
            {
                "text_id": row["text_id"],
                "outcome": outcome,
                "similarity": round(similarity, 6),
            }
        )
    mismatched = sum(s["outcome"] == "mismatched" for s in samples)
    return {
        "sampled": len(samples),
        "matched": len(samples) - mismatched,
        "mismatched": mismatched,
        "min_similarity": min((s["similarity"] for s in samples), default=None),
        "verdict": "fail" if mismatched else "pass",
        "reason": "vectors_mismatch" if mismatched else "ok",
        "samples": samples,
    }


class EmbedValidator(TaskChecker):
    """Recomputes a sample of each embed task's vectors with a reference model."""

    kinds = ("embed",)

    def __init__(
        self,
        api,
        http,
        references: dict[str, EmbeddingClient],
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
            upload = await self.download(job["download_url"], task_id)
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
        result, vectors = check_rows(inputs, upload, reference.model.dims)
        if "verdict" not in result:
            picked = pick_samples(inputs, job["seed"])
            try:
                expected = await reference.embed([row["text"] for row in picked])
            except Exception as exc:
                log.warning("task=%s reference failed: %s", task_id, exc)
                self.provider_failed()
                self.defer(task_id)
                return {}
            self.provider_worked()
            result |= compare(picked, vectors, expected)

        await self.submit_verdict(task_id, result)
        self.note_verdict(job, result)
        log.info(
            "task=%s miner=%s embed %d texts, matched %d/%d (min %s) verdict=%s reason=%s",
            task_id,
            job["miner"][:10],
            result["returned"],
            result.get("matched", 0),
            result.get("sampled", 0),
            result.get("min_similarity"),
            result["verdict"],
            result["reason"],
        )
        return result
