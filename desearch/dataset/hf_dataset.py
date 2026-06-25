import hashlib
import json
import os
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import List, Optional

import bittensor as bt

NEWS_REPO = "desearch/dataset"
CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "desearch-questions")
REFRESH_SECONDS = 4 * 60 * 60

REQUIRED_FIELDS = ("id", "question")

DATASET_LIMIT = 5000

DATASETS = (
    {
        "repo": "sentence-transformers/squad",
        "question_col": "question",
        "lane": "squad",
    },
    {
        "repo": "sentence-transformers/natural-questions",
        "question_col": "query",
        "lane": "nq",
    },
)


class HFQuestionPool:
    def __init__(
        self,
        repo_id: str = NEWS_REPO,
        cache_dir: str = CACHE_DIR,
        local_dir: Optional[str] = None,
        refresh_seconds: int = REFRESH_SECONDS,
        datasets=DATASETS,
        limit: int = DATASET_LIMIT,
    ):
        self.repo_id = repo_id
        self.cache_dir = Path(cache_dir)
        self.local_dir = Path(local_dir) if local_dir else None
        self.refresh_seconds = refresh_seconds
        self.datasets = datasets
        self.limit = limit

        self._rows: List[dict] = []
        self._loaded_at: float = 0.0

    def sample(self, n: int) -> Optional[List[dict]]:
        rows = self._ensure_rows()
        if not rows:
            return None

        by_lane: dict[str, List[dict]] = defaultdict(list)
        for row in rows:
            by_lane[row.get("lane") or "news"].append(row)
        for lane_rows in by_lane.values():
            random.shuffle(lane_rows)

        lanes = list(by_lane)
        random.shuffle(lanes)
        cursors = {lane: 0 for lane in lanes}

        out: List[dict] = []
        while len(out) < n:
            advanced = False
            for lane in lanes:
                if cursors[lane] < len(by_lane[lane]):
                    out.append(by_lane[lane][cursors[lane]])
                    cursors[lane] += 1
                    advanced = True
                    if len(out) >= n:
                        break
            if not advanced:
                break

        return out

    def _ensure_rows(self) -> List[dict]:
        fresh = time.time() - self._loaded_at < self.refresh_seconds
        if self._rows and fresh:
            return self._rows

        try:
            rows = self._load_rows()
        except Exception as e:
            bt.logging.error(f"[HFQuestionPool] Failed to load questions: {e}")
            rows = []

        if rows:
            self._rows = rows
            self._loaded_at = time.time()

        return self._rows

    def _load_rows(self) -> List[dict]:
        if self.local_dir is not None:
            rows = self._parse_files(sorted(self.local_dir.glob("*.jsonl")))
            return self._dedup_by_id(rows)

        try:
            paths = self._sync_from_hf()
        except Exception as e:
            bt.logging.warning(
                f"[HFQuestionPool] HF sync failed, falling back to cache: {e}"
            )
            paths = []
        if not paths:
            paths = sorted(self.cache_dir.rglob("*.jsonl"))

        rows = self._parse_files(paths) + self._load_datasets()
        return self._dedup_by_id(rows)

    @staticmethod
    def _dedup_by_id(rows: List[dict]) -> List[dict]:
        seen: set[str] = set()
        out: List[dict] = []
        for row in rows:
            if row["id"] in seen:
                continue
            seen.add(row["id"])
            out.append(row)
        return out

    def _sync_from_hf(self) -> List[Path]:
        from huggingface_hub import HfApi, hf_hub_download

        api = HfApi()
        files = api.list_repo_files(self.repo_id, repo_type="dataset")
        question_files = [
            f for f in files if f.startswith("questions/") and f.endswith(".jsonl")
        ]

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        paths: List[Path] = []
        for f in question_files:
            local = hf_hub_download(
                self.repo_id,
                f,
                repo_type="dataset",
                local_dir=str(self.cache_dir),
            )
            paths.append(Path(local))

        return paths

    def _parse_files(self, paths: List[Path]) -> List[dict]:
        rows: List[dict] = []
        for path in paths:
            for line in path.read_text().splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if all(row.get(k) for k in REQUIRED_FIELDS):
                    row.setdefault("start_date", None)
                    row.setdefault("end_date", None)
                    row.setdefault("lane", "news")
                    rows.append(row)

        return rows

    def _load_datasets(self) -> List[dict]:
        rows: List[dict] = []
        for cfg in self.datasets:
            try:
                rows += self._load_dataset(cfg)
            except Exception as e:
                bt.logging.error(f"[HFQuestionPool] Failed dataset {cfg['repo']}: {e}")
        return rows

    def _load_dataset(self, cfg: dict) -> List[dict]:
        from datasets import load_dataset

        ds = load_dataset(cfg["repo"], split="train")
        questions = ds[cfg["question_col"]]
        n = len(questions)
        lane = cfg["lane"]

        stride = max(1, n // self.limit)
        start = int(time.time() // self.refresh_seconds) % stride

        out: List[dict] = []
        seen: set[str] = set()
        for question in questions[start::stride]:
            question = (question or "").strip()
            if not question:
                continue
            qid = "h" + hashlib.sha1(question.encode("utf-8")).hexdigest()[:15]
            if qid in seen:
                continue
            seen.add(qid)
            out.append(
                {
                    "id": qid,
                    "question": question,
                    "start_date": None,
                    "end_date": None,
                    "lane": lane,
                }
            )
            if len(out) >= self.limit:
                break

        bt.logging.info(
            f"[HFQuestionPool] Loaded {len(out)} {lane} questions "
            f"from {cfg['repo']} (n={n}, stride={stride}, start={start})"
        )
        return out
