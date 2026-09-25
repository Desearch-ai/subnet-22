"""Search over /opt/unified: binary-256 dense candidates rescored with full vectors, BM25 chunk postings, z-score fusion, date boost."""

import json
import os
import re
import time
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import Stemmer

from .build import OUT, h64, terms

DIM = 256
ARMS = ["full", "head", "chunk", "bm_chunk", "bm_doc"]
MONTHS = {
    m: i + 1
    for i, m in enumerate(
        [
            "january",
            "february",
            "march",
            "april",
            "may",
            "june",
            "july",
            "august",
            "september",
            "october",
            "november",
            "december",
        ]
    )
}
QUARTER = re.compile(
    r"(?i)\b(?:(first|second|third|fourth)\s+quarter|q([1-4]))\s+(?:of\s+)?(?:fiscal\s+)?(20\d\d)\b"
)
MONTH_YEAR = re.compile(
    r"(?i)\b(" + "|".join(MONTHS) + r")\s+(?:\d{1,2},?\s+)?(20\d\d)\b"
)
YEAR = re.compile(r"\b(20[12]\d)\b")
ORD = {"first": 1, "second": 2, "third": 3, "fourth": 4}


def date_window(q: str, years: bool = False):
    """Publication window a question implies; quarter results are reported after the quarter ends."""
    if m := QUARTER.search(q):
        n = ORD[m.group(1).lower()] if m.group(1) else int(m.group(2))
        y = int(m.group(3))
        start = date(y, 3 * n - 2, 1)
        return start + timedelta(days=60), start + timedelta(days=92 + 75)
    if m := MONTH_YEAR.search(q):
        y, mo = int(m.group(2)), MONTHS[m.group(1).lower()]
        start = date(y, mo, 1)
        return start - timedelta(days=10), start + timedelta(days=31)
    if years and (m := YEAR.search(q)):
        y = int(m.group(1))
        return date(y, 1, 1), date(y, 12, 31)
    return None


def unit(x: np.ndarray) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12)


def prefix(q: np.ndarray, dim: int) -> np.ndarray:
    """The query cut to a corpus stored at fewer dimensions, renormalised so cosines hold."""
    return q if dim >= len(q) else unit(q[:dim])


def top(scores: np.ndarray, k: int) -> np.ndarray:
    """Indices of the k best scores, or all of them in a segment smaller than k."""
    if len(scores) <= k:
        return np.arange(len(scores))
    return np.argpartition(-scores, k)[:k]


def built_at(root: Path) -> int:
    return (root / "docs.jsonl").stat().st_mtime_ns


def sign(x) -> np.ndarray:
    return np.where(np.asarray(x) > 0, 1.0, -1.0).astype(np.float32)


class Unified:
    def __init__(
        self,
        root: Path = OUT,
        cand_doc: int = int(os.environ.get("UNIFIED_CAND_DOC", "200")),
        cand_chunk: int = int(os.environ.get("UNIFIED_CAND_CHUNK", "1000")),
    ):
        self.root = root
        self.built = built_at(root)
        self.cand_doc, self.cand_chunk = cand_doc, cand_chunk
        self.meta, self.offsets = [], []
        with open(root / "docs.jsonl", "rb") as fh:
            pos = 0
            for line in fh:
                d = json.loads(line)
                d.pop("text")
                self.meta.append(d)
                self.offsets.append(pos)
                pos += len(line)
        self.docs_path = root / "docs.jsonl"
        self.key_ix = {d["key"]: i for i, d in enumerate(self.meta)}
        self.pub = np.array(
            [
                date.fromisoformat(d["published"])
                if d["published"]
                else date(1970, 1, 1)
                for d in self.meta
            ]
        )
        self.head = np.load(root / "head.npy")
        self.full = np.load(root / "full.npy")
        self.head_c, self.full_c = sign(self.head[:, :DIM]), sign(self.full[:, :DIM])
        coarse = np.load(root / "chunk_coarse.npy", mmap_mode="r")
        self.chunk_c = np.empty(coarse.shape, dtype=np.float32)
        for i in range(0, len(coarse), 500_000):
            self.chunk_c[i : i + 500_000] = sign(coarse[i : i + 500_000])
        with open(root / "files.json") as fh:
            self.files = [np.load(f, mmap_mode="r") for f in json.load(fh)]
        self.cfile = np.load(root / "chunk_file.npy")
        self.crow = np.load(root / "chunk_row.npy")
        self.owner = np.load(root / "chunk_owner.npy")
        self.vocab = np.load(root / "bm_vocab.npy")
        self.indptr = np.load(root / "bm_indptr.npy")
        self.indices = np.load(root / "bm_indices.npy", mmap_mode="r")
        self.data = np.load(root / "bm_data.npy", mmap_mode="r")
        self.stem, self.cache = Stemmer.Stemmer("english"), {}

    def is_current(self) -> bool:
        """False once the directory has been rebuilt under this object."""
        return built_at(self.root) == self.built

    def text(self, i: int) -> str:
        with open(self.docs_path, "rb") as fh:
            fh.seek(self.offsets[i])
            return json.loads(fh.readline())["text"]

    def _dense_doc(self, q, qc, coarse, exact):
        cand = top(coarse @ qc, self.cand_doc)
        return cand, exact[cand].astype(np.float32) @ prefix(q, exact.shape[1])

    def _rescore_chunks(self, rows: np.ndarray, q: np.ndarray) -> np.ndarray:
        out = np.empty(len(rows), dtype=np.float32)
        f, r = self.cfile[rows], self.crow[rows]
        for fi in np.unique(f):
            s = np.where(f == fi)[0]
            s = s[np.argsort(r[s])]
            v = np.asarray(self.files[fi][r[s]], dtype=np.float32)
            out[s] = (v @ prefix(q, v.shape[1])) / np.linalg.norm(v, axis=1)
        return out

    def _per_doc(self, rows: np.ndarray, scores: np.ndarray):
        if rows.size == 0:
            return rows.astype(np.int64), scores.astype(np.float32)

        own = self.owner[rows]
        order = np.lexsort((-scores, own))
        own, scores = own[order], scores[order]
        first = np.r_[True, own[1:] != own[:-1]]
        return own[first], scores[first]

    def term_coverage(self, question: str) -> float:
        """Share of the question's terms the corpus knows at all."""
        wanted = set(terms(question, self.stem, self.cache))
        if not wanted:
            return 0.0
        return len(self._vocab_ids(wanted)) / len(wanted)

    def _vocab_ids(self, wanted: set) -> list[int]:
        ids = []
        for t in wanted:
            h = np.uint64(h64(t))
            i = int(np.searchsorted(self.vocab, h))
            if i < len(self.vocab) and self.vocab[i] == h:
                ids.append(i)
        return ids

    def _bm25(self, question: str) -> tuple[np.ndarray, np.ndarray]:
        ids = self._vocab_ids(set(terms(question, self.stem, self.cache)))
        scores = np.zeros(len(self.owner), dtype=np.float32)
        for i in ids:
            a, b = self.indptr[i], self.indptr[i + 1]
            scores[self.indices[a:b]] += self.data[a:b]
        rows = top(scores, self.cand_chunk)
        rows = rows[scores[rows] > 0]
        return self._per_doc(rows, scores[rows])

    def _bm25_doc(self, question: str) -> tuple[np.ndarray, np.ndarray]:
        """Page-level BM25: each term counts once, at its best passage, so terms may sit apart."""
        ids = self._vocab_ids(set(terms(question, self.stem, self.cache)))
        total = np.zeros(len(self.meta), dtype=np.float32)
        for i in ids:
            a, b = self.indptr[i], self.indptr[i + 1]
            best = np.zeros(len(self.meta), dtype=np.float32)
            np.maximum.at(best, self.owner[self.indices[a:b]], self.data[a:b])
            total += best
        docs = top(total, self.cand_doc)
        docs = docs[total[docs] > 0]
        return docs.astype(np.int64), total[docs]

    def arms(self, question: str, q: np.ndarray):
        """Per arm (doc ids, scores) plus stage timings in ms."""
        t, out = {}, {}
        qc = unit(q[:DIM])
        t0 = time.perf_counter()
        out["full"] = self._dense_doc(q, qc, self.full_c, self.full)
        out["head"] = self._dense_doc(q, qc, self.head_c, self.head)
        t["dense_doc"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        rows = top(self.chunk_c @ qc, self.cand_chunk)
        t["dense_chunk_scan"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        out["chunk"] = self._per_doc(rows, self._rescore_chunks(rows, q))
        t["dense_chunk_rescore"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        out["bm_chunk"] = self._bm25(question)
        t["bm25"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        out["bm_doc"] = self._bm25_doc(question)
        t["bm25_doc"] = time.perf_counter() - t0
        return out, {k: v * 1000 for k, v in t.items()}

    def fuse(
        self, out: dict, weights: dict, date_boost: float, win
    ) -> tuple[np.ndarray, np.ndarray]:
        return fuse(out, weights, date_boost, win, self.pub)


def zscores(sc: np.ndarray, top: int = 200) -> np.ndarray:
    head = np.sort(sc)[::-1][:top]
    return (sc - head.mean()) / (head.std() + 1e-6)


def fuse(
    out: dict, weights: dict, date_boost: float, win, pub: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Union of arm candidates with fused score; a doc missing from an arm gets that arm's floor."""
    cand = np.unique(np.concatenate([out[a][0] for a in ARMS]))
    pos = {d: i for i, d in enumerate(cand.tolist())}
    S = np.zeros(len(cand), dtype=np.float32)
    for a, w in weights.items():
        if not w:
            continue
        ids, sc = out[a]
        z = zscores(sc)
        col = np.full(len(cand), z.min() if len(z) else 0.0, dtype=np.float32)
        col[[pos[d] for d in ids.tolist()]] = z
        S += w * col
    if win and date_boost:
        lo, hi = win
        p = pub[cand]
        S += date_boost * ((p >= lo) & (p <= hi))
    order = np.argsort(-S)
    return cand[order], S[order]


class Index:
    """The built index and the segments synced since, searched as one; a newer copy of a page hides older ones."""

    def __init__(self, segments: list[Unified]):
        self.segments = segments
        sizes = [len(seg.meta) for seg in segments]
        self.starts = np.r_[0, np.cumsum(sizes)].astype(np.int64)
        self.meta = [d for seg in segments for d in seg.meta]
        self.pub = (
            np.concatenate([seg.pub for seg in segments])
            if segments
            else np.array([], dtype=object)
        )
        self.key_ix: dict[str, int] = {}
        self.hidden = np.zeros(len(self.meta), dtype=bool)
        for doc in range(len(self.meta) - 1, -1, -1):
            key = self.meta[doc]["key"]
            if key in self.key_ix:
                self.hidden[doc] = True
            else:
                self.key_ix[key] = doc

    def locate(self, doc: int) -> tuple[Unified, int]:
        at = int(np.searchsorted(self.starts, doc, side="right")) - 1
        return self.segments[at], doc - int(self.starts[at])

    def text(self, doc: int) -> str:
        segment, local = self.locate(doc)
        return segment.text(local)

    def term_coverage(self, question: str) -> float:
        return max((seg.term_coverage(question) for seg in self.segments), default=0.0)

    def arms(self, question: str, q: np.ndarray):
        out = {arm: ([], []) for arm in ARMS}
        timings: dict[str, float] = {}
        for segment, start in zip(self.segments, self.starts):
            found, spent = segment.arms(question, q)
            for arm, (ids, scores) in found.items():
                ids = np.asarray(ids, dtype=np.int64) + start
                keep = ~self.hidden[ids]
                out[arm][0].append(ids[keep])
                out[arm][1].append(np.asarray(scores, dtype=np.float32)[keep])
            for stage, ms in spent.items():
                timings[stage] = timings.get(stage, 0.0) + ms
        merged = {
            arm: (
                np.concatenate(ids) if ids else np.array([], dtype=np.int64),
                np.concatenate(scores) if scores else np.array([], dtype=np.float32),
            )
            for arm, (ids, scores) in out.items()
        }
        return merged, timings

    def fuse(
        self, out: dict, weights: dict, date_boost: float, win
    ) -> tuple[np.ndarray, np.ndarray]:
        return fuse(out, weights, date_boost, win, self.pub)
