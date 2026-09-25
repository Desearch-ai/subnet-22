"""One index over every corpus (benchmark gold pages, news, TechCrunch) from the saved vectors; nothing is re-embedded."""

from __future__ import annotations

import hashlib
import json
import multiprocessing as mp
import os
import re
import sys
import time
from array import array
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from urllib.parse import unquote, urlsplit

import numpy as np
import Stemmer
from bm25s.stopwords import STOPWORDS_EN

from .chunking import para_chunks

OUT = Path(os.environ.get("UNIFIED_DIR", "/opt/unified"))
# Synced segments and the vectors they point at live beside the index, so a rebuild can swap it.
LIVE = Path(os.environ.get("UNIFIED_LIVE", f"{OUT}_live"))
STORE = Path(os.environ.get("UNIFIED_STORE", f"{OUT}_store"))
# Pages and queries must be embedded by the same model; see desearch/embedding.py.
MODEL = os.environ.get("ENGINE_EMBED_MODEL", "qwen3-embedding-8b")
DIM, SHARD, K1, B = 256, 8192, 1.2, 0.75
TOK = re.compile(r"[a-z0-9]+")
STOP = set(STOPWORDS_EN)
WORKERS = 8


def canon(url: str) -> str:
    s = urlsplit(url.strip())
    host = s.netloc.lower().removeprefix("www.").removeprefix("m.")
    host = host.replace(".m.wikipedia.org", ".wikipedia.org").replace(
        "bbc.co.uk", "bbc.com"
    )
    path = unquote(s.path).rstrip("/") or "/"
    return host + path + ("?" + s.query if s.query else "")


def terms(text: str, stem, cache: dict) -> list[str]:
    out = []
    for w in TOK.findall(text.lower()):
        if len(w) < 2 or w in STOP:
            continue
        s = cache.get(w)
        if s is None:
            s = cache[w] = stem.stemWord(w)
        out.append(s)
    return out


def iso_day(value) -> str:
    """YYYY-MM-DD, or empty when the source date does not parse."""
    try:
        return date.fromisoformat((value or "")[:10]).isoformat()
    except ValueError:
        return ""


def h64(s: str) -> int:
    return int.from_bytes(hashlib.blake2b(s.encode(), digest_size=8).digest(), "little")


def flat_source(docs_path: Path, emb_dir: Path, files: list, keep: set | None = None):
    """index_bench layout: head, full, chunks per non-empty doc, 8192 rows per shard."""
    shard_files = sorted(p for p in emb_dir.glob("[0-9]*.npy") if "tmp" not in p.name)
    base = len(files)
    files += [str(p) for p in shard_files]
    k = 0
    with open(docs_path) as fh:
        lines = list(fh)
    for line in lines:
        d = json.loads(line)
        if not d["text"]:
            continue
        n = 2 + len(para_chunks(d["text"]))
        if keep is None or d["url"] in keep:
            yield d, [(base + (k + i) // SHARD, (k + i) % SHARD) for i in range(n)]
        k += n


def sharded_source(root: Path, files: list, only: set | None = None):
    """news_pipeline layout: one shard per .done file, docs in .done order, head/full/chunks each."""
    by_url = {}
    with open(root / "docs.jsonl") as fh:
        for line in fh:
            d = json.loads(line)
            by_url[d["url"]] = d
    skipped = 0
    for f in sorted(root.glob("emb/*.npy")):
        done = f.with_suffix(".done")
        if not done.exists():
            continue
        ds = [by_url[u] for u in done.read_text().split()]
        sizes = [2 + len(para_chunks(d["text"])) for d in ds]
        if sum(sizes) != np.load(f, mmap_mode="r").shape[0]:
            skipped += 1
            continue
        fi = len(files)
        files.append(str(f))
        pos = 0
        for d, n in zip(ds, sizes):
            if only is None or d["url"] in only:
                yield d, [(fi, pos + i) for i in range(n)]
            pos += n
    if skipped:
        print(
            f"  {root}: skipped {skipped} shards whose row count no longer matches docs",
            flush=True,
        )


def bm25_part(args):
    out, lo, hi, part = args
    stem, cache, hcache = Stemmer.Stemmer("english"), {}, {}
    rows, hs, tfs, lens = array("i"), array("Q"), array("H"), array("i")
    with open(out / "chunks.txt") as fh:
        for i, line in enumerate(fh):
            if i < lo:
                continue
            if i >= hi:
                break
            ts = terms(line, stem, cache)
            lens.append(len(ts))
            for t, c in Counter(ts).items():
                h = hcache.get(t)
                if h is None:
                    h = hcache[t] = h64(t)
                rows.append(i)
                hs.append(h)
                tfs.append(min(c, 65535))
    for name, a, dt in (
        ("rows", rows, np.int32),
        ("hs", hs, np.uint64),
        ("tfs", tfs, np.uint16),
        ("lens", lens, np.int32),
    ):
        np.save(out / f"bm_{part}_{name}.npy", np.frombuffer(a, dtype=dt))
    return part


@dataclass
class CorpusStats:
    """A built index's term counts, so a small segment scores on the same scale as it."""

    vocab: np.ndarray
    df: np.ndarray
    chunks: int
    avgdl: float | None

    @classmethod
    def of(cls, root: Path) -> CorpusStats:
        stats = root / "bm_stats.json"
        saved = json.loads(stats.read_text()) if stats.exists() else {}
        return cls(
            vocab=np.load(root / "bm_vocab.npy"),
            df=np.diff(np.load(root / "bm_indptr.npy")),
            chunks=len(np.load(root / "chunk_owner.npy", mmap_mode="r")),
            avgdl=saved.get("avgdl"),
        )

    def idf(self, vocab: np.ndarray, df: np.ndarray) -> np.ndarray:
        """Known terms take this corpus's frequency; new ones their own, against its size."""
        at = np.minimum(np.searchsorted(self.vocab, vocab), len(self.vocab) - 1)
        known = self.vocab[at] == vocab
        df = np.where(known, self.df[at], df)
        return np.log(1 + (self.chunks - df + 0.5) / (df + 0.5)).astype(np.float32)


def build_bm25(out: Path, n_chunks: int, corpus: CorpusStats | None = None) -> None:
    workers = min(WORKERS, max(1, n_chunks // 100_000))
    step = (n_chunks + workers - 1) // workers
    jobs = [(out, i * step, min(n_chunks, (i + 1) * step), i) for i in range(workers)]
    if workers == 1:
        bm25_part(jobs[0])
    else:
        with mp.get_context("fork").Pool(workers) as pool:
            list(pool.imap_unordered(bm25_part, jobs))

    def load(name):
        return np.concatenate(
            [np.load(out / f"bm_{i}_{name}.npy") for i in range(workers)]
        )

    lens = load("lens").astype(np.float32)
    rows, tfs = load("rows"), load("tfs").astype(np.float32)
    vocab, inv = np.unique(load("hs"), return_inverse=True)
    df = np.bincount(inv, minlength=len(vocab))
    if corpus is None:
        idf = np.log(1 + (n_chunks - df + 0.5) / (df + 0.5)).astype(np.float32)
        avgdl = float(lens.mean())
    else:
        idf = corpus.idf(vocab, df)
        avgdl = corpus.avgdl or float(lens.mean())
    norm = K1 * (1 - B + B * lens / avgdl)
    w = idf[inv] * tfs * (K1 + 1) / (tfs + norm[rows])
    del tfs
    order = np.argsort(inv, kind="stable")
    del inv
    np.save(out / "bm_indices.npy", rows[order])
    np.save(out / "bm_data.npy", w[order].astype(np.float32))
    np.save(out / "bm_indptr.npy", np.r_[0, np.cumsum(df)].astype(np.int64))
    np.save(out / "bm_vocab.npy", vocab)
    (out / "bm_stats.json").write_text(
        json.dumps({"chunks": n_chunks, "avgdl": float(lens.mean())})
    )
    for p in out.glob("bm_[0-9]*_*.npy"):
        p.unlink()
    print(f"bm25: {len(vocab):,} terms, {len(rows):,} postings", flush=True)


def week_urls() -> set:
    path = Path("/opt/week/articles.jsonl")
    return {json.loads(line)["url"] for line in open(path)} if path.exists() else set()


def ready_segments(live: Path = LIVE) -> list[Path]:
    """Finished live segments, oldest first."""
    return sorted(p.parent for p in live.glob("*/READY"))


def segment_source(segment: Path, files: list):
    """A live segment: its store shard holds head, full and passages per page, in docs order."""
    with open(segment / "files.json") as fh:
        (shard,) = json.load(fh)
    fi = len(files)
    files.append(shard)
    pos = 0
    with open(segment / "docs.jsonl") as fh:
        for line in fh:
            d = json.loads(line)
            n = 2 + len(para_chunks(d["text"]))
            yield d, [(fi, pos + i) for i in range(n)]
            pos += n


def write_index(
    out: Path,
    sources: list[tuple[str, Iterable]],
    files: list[str],
    corpus: CorpusStats | None = None,
) -> int:
    """One index directory from (doc, vector rows) sources; the first copy of a URL wins."""
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    seen, dups, counts = set(), Counter(), Counter()
    head_p, full_p, owner = [], [], array("i")
    chunk_f, chunk_r = array("i"), array("i")
    with open(out / "docs.jsonl", "w") as fd, open(out / "chunks.txt", "w") as fc:
        for name, gen in sources:
            for d, rows in gen:
                key = canon(d["url"])
                if key in seen:
                    dups[name] += 1
                    continue
                seen.add(key)
                j = len(head_p)
                head_p.append(rows[0])
                full_p.append(rows[1])
                for c, (f, r) in zip(para_chunks(d["text"]), rows[2:]):
                    chunk_f.append(f)
                    chunk_r.append(r)
                    owner.append(j)
                    fc.write((d["title"] + " " + c).replace("\n", " ") + "\n")
                meta = {
                    "url": d["url"],
                    "key": key,
                    "title": d.get("title") or "",
                    "published": iso_day(d.get("published")),
                    "source": name,
                    "text": d["text"],
                }
                fd.write(json.dumps(meta, ensure_ascii=False) + "\n")
                counts[name] += 1
    n_docs, n_chunks = len(head_p), len(owner)
    if not n_docs:
        raise ValueError("nothing to index")
    print(
        f"docs {dict(counts)} (duplicates dropped {dict(dups)}), chunks {n_chunks:,} ({time.time() - t0:.0f}s)",
        flush=True,
    )

    with open(out / "files.json", "w") as fh:
        json.dump(files, fh)
    cf, cr = np.frombuffer(chunk_f, np.int32), np.frombuffer(chunk_r, np.int32)
    np.save(out / "chunk_file.npy", cf.astype(np.int16))
    np.save(out / "chunk_row.npy", cr)
    np.save(out / "chunk_owner.npy", np.frombuffer(owner, np.int32))
    hp, fp = np.array(head_p, np.int32), np.array(full_p, np.int32)

    # Shards may store a prefix of the 4096 dimensions; document vectors keep the same width.
    width = min(4096, np.load(files[0], mmap_mode="r").shape[1])
    head = np.lib.format.open_memmap(
        out / "head.npy", "w+", np.float16, (n_docs, width)
    )
    full = np.lib.format.open_memmap(
        out / "full.npy", "w+", np.float16, (n_docs, width)
    )
    coarse = np.lib.format.open_memmap(
        out / "chunk_coarse.npy", "w+", np.float16, (n_chunks, DIM)
    )
    for fi, path in enumerate(files):
        m = np.load(path, mmap_mode="r")
        for pf, pr, dst, dim in (
            (hp[:, 0], hp[:, 1], head, width),
            (fp[:, 0], fp[:, 1], full, width),
            (cf, cr, coarse, DIM),
        ):
            sel = np.where(pf == fi)[0]
            if not len(sel):
                continue
            v = np.asarray(m[pr[sel]], dtype=np.float32)[:, :dim]
            v /= np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-12)
            dst[sel] = v.astype(np.float16)
        if fi % 100 == 0:
            print(
                f"  vectors: file {fi}/{len(files)} ({time.time() - t0:.0f}s)",
                flush=True,
            )
    del head, full, coarse
    print(f"vectors done ({time.time() - t0:.0f}s)", flush=True)

    build_bm25(out, n_chunks, corpus)
    print(f"DONE ({time.time() - t0:.0f}s)")
    return n_docs


def main() -> None:
    files: list[str] = []
    keep_path = os.environ.get("HARVEST_KEEP")
    keep = set(Path(keep_path).read_text().split()) if keep_path else None
    segments = ready_segments()
    sources = [
        # Newest first: a page synced since the last build replaces its older copy.
        *(("live", segment_source(seg, files)) for seg in reversed(segments)),
        (
            "bench",
            flat_source(
                Path("/opt/bench/docs_add.jsonl"), Path("/opt/bench/emb8b_add"), files
            ),
        ),
        (
            "bench",
            flat_source(
                Path("/opt/bench/docs_v1.jsonl"), Path("/opt/bench/emb8b"), files
            ),
        ),
        (
            "bcplus",
            flat_source(
                Path(os.environ.get("BCPLUS_DOCS", "/opt/bcplus/docs.jsonl")),
                Path(os.environ.get("BCPLUS_EMB", "/opt/bcplus/emb")),
                files,
            ),
        ),
        *(
            ("harvest", flat_source(root / "docs.jsonl", root / "emb", files, keep))
            for root in sorted(
                Path(os.environ.get("HARVEST", "/opt/harvest")).glob("*/")
            )
            if (root / "docs.jsonl").exists()
        ),
        ("techcrunch", sharded_source(Path("/opt/tc"), files)),
        ("news", sharded_source(Path("/opt/news"), files)),
        ("week", sharded_source(Path("/opt/news"), files, week_urls())),
    ]
    wanted = (sys.argv[1] if len(sys.argv) > 1 else "bench,techcrunch,news").split(",")
    sources = [(name, gen) for name, gen in sources if name in wanted]
    write_index(OUT, sources, files)
    if "live" in wanted:
        # Segments folded in here are skipped by the service until they are deleted.
        (OUT / "merged.json").write_text(json.dumps([seg.name for seg in segments]))


if __name__ == "__main__":
    main()
