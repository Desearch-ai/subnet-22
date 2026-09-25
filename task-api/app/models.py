from __future__ import annotations

from collections import Counter
from typing import Literal, get_args

from pydantic import BaseModel, ConfigDict, Field, model_validator

from . import rounds

MAX_URL_DETAILS = 1000
Outcome = Literal[
    "matched",
    "mismatched",
    "unverifiable",
    "errors_confirmed",
    "errors_unconfirmed",
    "not_fetched",
]


class CompleteBody(BaseModel):
    key: str
    rows: int = Field(0, ge=0)
    ok: int = Field(0, ge=0)
    errors: int = Field(0, ge=0)
    bytes: int = Field(0, ge=0)


class Sample(BaseModel):
    model_config = ConfigDict(extra="allow")

    url: str
    outcome: Outcome
    similarity: float = 0.0
    miner_status: int = 0
    validator_status: int = 0
    miner_chars: int = 0
    validator_chars: int = 0


class UrlDetail(BaseModel):
    model_config = ConfigDict(extra="forbid")

    url: str = Field(max_length=4096)
    status: int = 0
    error: str | None = Field(None, max_length=64)
    text_chars: int = 0
    sampled: bool = False
    outcome: str | None = Field(None, max_length=32)
    why: str | None = Field(None, max_length=500)
    similarity: float | None = None
    precision: float | None = None
    recall: float | None = None
    growth: float | None = None
    validator_error: str | None = Field(None, max_length=200)
    via: str = Field("", max_length=16)
    miner_chars: int | None = None
    validator_chars: int | None = None
    miner_snippet: str | None = Field(None, max_length=500)
    validator_snippet: str | None = Field(None, max_length=500)
    diff_at: int | None = None
    miner_window: str | None = Field(None, max_length=500)
    validator_window: str | None = Field(None, max_length=500)
    rejected: bool = False


class Score(BaseModel):
    model_config = ConfigDict(extra="allow")

    returned: int = Field(0, ge=0)
    missing: int = Field(0, ge=0)
    duplicates: int = Field(0, ge=0)
    error_rows: int = Field(0, ge=0)
    sampled: int = Field(0, ge=0)
    matched: int = Field(0, ge=0)
    mismatched: int = Field(0, ge=0)
    unverifiable: int = Field(0, ge=0)
    not_fetched: int = Field(0, ge=0)
    errors_confirmed: int = Field(0, ge=0)
    errors_unconfirmed: int = Field(0, ge=0)
    reextract_mismatch: int = Field(0, ge=0)
    verdict: Literal["pass", "fail", "void"]
    reason: str = ""
    samples: list[Sample] = []
    urls: list[UrlDetail] = Field([], max_length=MAX_URL_DETAILS)
    rejected: list[str] = Field([], max_length=MAX_URL_DETAILS)

    @model_validator(mode="after")
    def counts_match_samples(self) -> Score:
        counts = Counter(sample.outcome for sample in self.samples)
        if self.sampled != len(self.samples) or any(
            getattr(self, outcome) != counts[outcome] for outcome in get_args(Outcome)
        ):
            raise ValueError("the counts do not match the samples")
        if self.error_rows > self.returned:
            raise ValueError("error_rows cannot exceed returned")
        return self


class ClaimBody(BaseModel):
    kind: Literal["crawl", "embed"] = "crawl"


class OpenBody(BaseModel):
    kinds: list[Literal["crawl", "embed"]] = Field(
        ["crawl"], min_length=1, max_length=2
    )
    skip: list[str] = Field([], max_length=64)


class EmbedSample(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text_id: str = Field(max_length=300)
    outcome: Literal["matched", "mismatched", "unverifiable"]
    similarity: float | None = None


class EmbedScore(BaseModel):
    model_config = ConfigDict(extra="forbid")

    returned: int = Field(0, ge=0)
    missing: int = Field(0, ge=0)
    duplicates: int = Field(0, ge=0)
    malformed: int = Field(0, ge=0)
    sampled: int = Field(0, ge=0)
    matched: int = Field(0, ge=0)
    mismatched: int = Field(0, ge=0)
    unverifiable: int = Field(0, ge=0)
    min_similarity: float | None = None
    verdict: Literal["pass", "fail", "void"]
    reason: str = Field("", max_length=64)
    samples: list[EmbedSample] = Field([], max_length=MAX_URL_DETAILS)

    @model_validator(mode="after")
    def counts_match_samples(self) -> EmbedScore:
        counts = Counter(sample.outcome for sample in self.samples)
        if self.sampled != len(self.samples) or any(
            getattr(self, outcome) != counts[outcome]
            for outcome in ("matched", "mismatched", "unverifiable")
        ):
            raise ValueError("the counts do not match the samples")
        return self


class Release(BaseModel):
    reason: Literal["missing"] = "missing"


class Enqueue(BaseModel):
    urls: list[rounds.Url]
    batch_target: int = Field(rounds.BATCH_TARGET, gt=0)
